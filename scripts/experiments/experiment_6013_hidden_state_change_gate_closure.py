#!/usr/bin/env python3
"""Exp 6013 -- REQ-ARC-WMTE-6013: close the change gate's HIDDEN-STATE coverage hole.

WHAT THIS MEASURES, AND WHY IT IS NOT exp6012 AGAIN
===================================================
exp6012 MEASURED a hole and stopped there: its own `acceptance_gate_passed` is FALSE,
because its finding is "the live hidden-state gate admits a spurious writer". It changed
no production code. This experiment measures the REPAIR: the hidden-state admission
decision is now routed through the same symmetric union-fidelity `change_gate_decision`
the plain branch uses, behind CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE (default-off, and
defaulting to follow REQ-6011's own flag so the gate arm is not a silent no-op on the 11
wall games).

The distinction matters for what a reader may conclude. exp6012's rows are a
RECONSTRUCTION of the two gates by an analysis harness. These rows are the PRODUCTION
path: every number here comes out of `select_trusted_world_model`, the function
E3AgentPolicy actually calls, reading the `change_gate` record that function now attaches
to every CandidateScore. That is deliberate -- this project has twice shipped a guard that
did not fire on its own origin incident, and both times the guard had been validated
against a reimplementation of the thing it was meant to guard rather than against the
thing itself.

THE FIVE QUESTIONS, AND WHY EACH ONE IS LOAD-BEARING
====================================================
1. DOES IT FIRE ON THE ORIGIN INCIDENT? The origin degenerates are the REAL files
   results/arc_e3/ft09/world_model.py (12 bare `return grid` branches) and
   results/arc_e3/lp85/world_model.py. They are loaded through the production
   `e3.load_engine` and routed through the production hidden-state function. Note these
   two games are not themselves in HIDDEN_STATE_GAME_IDS -- that is exactly why they are
   run here: `select_trusted_world_model` is game-agnostic, so putting the origin
   incident's own engines through the repaired path is the sharpest available test of
   "would this gate have caught the thing it exists to catch". A gate that only rejects
   engines constructed to be rejected has proved nothing.

2. DOES IT CLOSE THE MEASURED HOLE? exp6012's `base_plus_spurious` -- correct on every
   real change, plus one write to a provably mask-free cell -- must be rejected on rows
   where the incumbent `trust_pass` admits it.

3. DOES IT ADMIT A GENUINELY GOOD ENGINE? A gate that rejects everything is not an
   improvement over a gate that admits identity engines. The hand-written dc22 navigation
   engine (imported from exp6011, never re-typed) is the must-not-fire control.

4. IS THE DEFAULT REALLY INERT? With both flags unset, the resolved decision must be the
   incumbent's, on every row. This is the claim the operator is relying on when told the
   change ships default-off, and it is checked rather than asserted.

5. IS THE PASS REGION NON-EMPTY, AND COULD EACH GATE HAVE GONE THE OTHER WAY? A pass that
   could not have failed is not evidence. Every acceptance gate below is paired with the
   population it was computed over, and the run refuses if any of them was decided over an
   empty set.

SUBSTRATE: no GPU, no model, no LLM. Pure numpy over cached offline-arcade transitions
(`e3.collect_transitions`). `inference_substrate` is
`verifier_ensemble_against_cached_candidates` accordingly, whose duration floor is 1s.
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
    try:
        return str(pathlib.Path(p).relative_to(REPO))
    except ValueError:
        return str(p)


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
            "CARNOT_ARC_E3_DIR=$PWD/results/arc_e3_origin_fixtures .venv/bin/python "
            + str(pathlib.Path(__file__).resolve().relative_to(REPO))
        ),
    }


sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_world_model_trust_energy as te  # noqa: E402

# Imported, never re-typed. Two independent reimplementations of one formula agreeing with
# each other is not evidence about the system -- this project has a recorded incident where
# exactly that produced 44/44 agreement on a formula that was wrong in both copies.
from experiment_6012_hidden_state_trust_gate_hole import (  # noqa: E402
    _frame_hud_mask,
    _make_arms,
)

OUT = REPO / "results/experiment_6013_hidden_state_change_gate_closure.json"

# The two REAL on-disk engines named in GAP-WM-TRUST-GATE as the origin incident.
ORIGIN_INCIDENT_GAMES = ("ft09", "lp85")
# The game whose dynamics are known well enough from ops/arc_solve_registry.yaml to
# hand-write a correct engine; exp6011 owns that definition.
HANDWRITTEN_GAME = "dc22"


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _decide(transitions, engine, *, hud_mask, hidden_state=True, hud_mask_enabled=None) -> dict:
    """One PRODUCTION hidden-state decision for one engine.

    Calls `select_trusted_world_model` exactly as E3AgentPolicy's hidden-state branch does
    -- the WHOLE transition list, not a pre-split tail. The function splits internally, so a
    caller that pre-split would have the tail split AGAIN and would score the last ~1/9 of
    the corpus; that mistake rejected even a perfect engine in an earlier harness and is the
    reason the change-gate computation now lives inside the function that owns the split.
    """

    sel = te.select_trusted_world_model(
        list(transitions),
        [
            te.WorldModelCandidate(
                name="candidate", engine=engine, is_level_complete=lambda g: False
            )
        ],
        hidden_state=hidden_state,
        hud_mask=hud_mask,
        # CORRIGENDUM 2026-07-27: SUPPLYING a mask is not ENABLING one. Passing `hud_mask`
        # alone left `WorldModelVerifier` to consult the module flag, which is default-off,
        # so the original run's entire mask arm ran unmasked. Passing the resolved boolean
        # explicitly is what makes the arm the arm it claims to be.
        hud_mask_enabled=hud_mask_enabled,
    )
    score = sel.selected_score
    cg = dict(score.change_gate)
    return {
        # --- the INCUMBENT decision (recall-only consistency) --------------------
        "incumbent_trust_pass": bool(score.trust_pass),
        "incumbent_consistency": round(float(score.heldout_change_consistency), 6),
        "incumbent_nondegenerate": bool(score.nondegenerate),
        # --- the REPAIRED decision (symmetric union fidelity) --------------------
        "change_gate_pass": bool(score.change_gate_pass),
        "change_gate_reason": str(cg.get("reason", "unavailable")),
        "change_fidelity": cg.get("change_fidelity"),
        "spurious_changed_cells": cg.get("spurious_changed_cells"),
        "correct_changed_cells": cg.get("correct_changed_cells"),
        "noop_hallucination_rate": cg.get("noop_hallucination_rate"),
        "noop_channel_measurable": cg.get("noop_channel_measurable"),
        "noop_ok_is_vacuous": cg.get("noop_ok_is_vacuous"),
        "invented_changed_cells": cg.get("invented_changed_cells"),
        "invented_change_rate": cg.get("invented_change_rate"),
        "n_changing": cg.get("n_changing"),
        "n_noop": cg.get("n_noop"),
        "hud_mask_status": cg.get("hud_mask_status"),
        "hud_mask_cells": cg.get("hud_mask_cells"),
        # The record must be populated in EVERY arm, including a control -- an empty dict
        # here would mean the four-arm matrix has nothing to compare on this row.
        "change_gate_record_populated": bool(cg),
    }


def _one(game: str, n: int, seed: int) -> dict:
    row: dict = {"game": game, "seed": seed, "n_requested": n}
    t0 = time.time()
    try:
        trans, cell = e3.collect_transitions(game, n=n, seed=seed)
    except Exception as exc:
        row["error"] = f"collect_transitions:{type(exc).__name__}:{exc!r}"[:300]
        return row
    row["n_transitions"] = len(trans)
    row["cell"] = int(cell)
    if len(trans) < 6:
        row["error"] = "too_few_transitions"
        return row

    frame_mask = _frame_hud_mask(game)
    mask = e3.logical_hud_mask(frame_mask, cell) if frame_mask is not None else None
    row["hud_mask_available"] = bool(mask is not None)

    arms, base_info = _make_arms(game, trans, mask)
    row.update(base_info)

    # The REAL on-disk engine, through the production loader, for the origin-incident games.
    if game in ORIGIN_INCIDENT_GAMES:
        try:
            ondisk, _done = e3.load_engine(game)
            arms["ondisk_real_engine"] = ondisk
            row["ondisk_loaded"] = True
        except Exception as exc:
            row["ondisk_error"] = f"load_engine:{type(exc).__name__}:{exc!r}"[:300]
            row["ondisk_loaded"] = False

    decisions: dict = {}
    for mask_on in (0, 1):
        active = mask if mask_on else None
        for name, engine in arms.items():
            decisions[f"mask={mask_on}|engine={name}"] = _decide(
                trans, engine, hud_mask=active, hud_mask_enabled=bool(mask_on)
            )
    row["decisions"] = decisions
    row["elapsed_s"] = round(time.time() - t0, 3)
    return row


def _games(limit: int | None) -> list[str]:
    """The 11 hidden-state games (the branch under repair) plus the origin-incident pair.

    The origin games are NOT hidden-state, and including them is the point: they are the
    engines the gap entry was written about, so routing them through the repaired path is
    what makes gate 1 a test of the incident rather than a test of a construction.
    """

    games = list(te.HIDDEN_STATE_GAME_IDS) + [
        g for g in ORIGIN_INCIDENT_GAMES if g not in te.HIDDEN_STATE_GAME_IDS
    ]
    if HANDWRITTEN_GAME not in games:
        games.append(HANDWRITTEN_GAME)
    return games[:limit] if limit else games


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    # PRECONDITIONS. This experiment loads no model and touches no GPU, so the only real
    # precondition is that the offline arcade corpus is reachable; the flags are recorded
    # because a run under a flipped flag would measure something else entirely.
    preconditions = [
        {
            "resource": "offline_arcade_environment_files",
            "available": (REPO / "environment_files").is_dir(),
        },
        {
            "resource": "ondisk_origin_incident_engines",
            "available": all(
                (REPO / f"results/arc_e3/{g}/world_model.py").exists()
                for g in ORIGIN_INCIDENT_GAMES
            ),
        },
        {
            "resource": "flags_default_off",
            "available": (
                e3.SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED is False
                and e3.SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED is False
                and e3.SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED is None
            ),
        },
    ]
    if not all(p["available"] for p in preconditions):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(
            json.dumps(
                {
                    "experiment": "experiment_6013_hidden_state_change_gate_closure",
                    "honest_verdict": "blocked_preconditions_unmet",
                    "preconditions_checked": preconditions,
                },
                indent=2,
            )
        )
        print("BLOCKED: preconditions unmet", preconditions)
        return 1

    analyser_t0 = time.time()
    rows: list[dict] = []
    for game in _games(args.limit):
        for seed in args.seeds:
            rows.append(_one(game, args.n, seed))
            print(f"  {game} seed={seed} done", flush=True)

    ok = [r for r in rows if "error" not in r and r.get("decisions")]

    def _sel(engine: str, mask_on: int, key: str, pred) -> list[str]:
        """Row ids where `pred` holds for one (engine, mask) arm. Returns the SET, not a
        count -- a total cannot be diffed against a control, and this project's own
        measurement discipline requires failure SETS."""

        out = []
        for r in ok:
            d = r["decisions"].get(f"mask={mask_on}|engine={engine}")
            if d is not None and pred(d.get(key)):
                out.append(f"{r['game']}@{r['seed']}")
        return sorted(out)

    def _support(engine: str, mask_on: int) -> list[str]:
        return _sel(engine, mask_on, "change_gate_record_populated", lambda v: True)

    # ---- GATE 1: fires on the ORIGIN INCIDENT -------------------------------------
    # A REJECTION IS NOT AUTOMATICALLY EVIDENCE. `no_changing_transitions` rejects EVERY
    # engine on that corpus, a perfect one included, so counting it as a catch would be a
    # pass that could not have failed. Each origin arm is therefore classified into
    # DISCRIMINATING (rejected for a reason that a better engine could have avoided, over a
    # non-empty changing population) versus VACUOUS, and only the discriminating set is
    # allowed to support the origin-incident claim.
    origin_rows = [r for r in ok if r["game"] in ORIGIN_INCIDENT_GAMES and r.get("ondisk_loaded")]
    origin_rejected, origin_admitted, origin_incumbent_admitted = [], [], []
    origin_discriminating, origin_vacuous = [], []
    for r in origin_rows:
        for mask_on in (0, 1):
            d = r["decisions"].get(f"mask={mask_on}|engine=ondisk_real_engine")
            if d is None:
                continue
            tag = f"{r['game']}@{r['seed']}|mask={mask_on}"
            (origin_rejected if not d["change_gate_pass"] else origin_admitted).append(tag)
            if d["incumbent_trust_pass"]:
                origin_incumbent_admitted.append(tag)
            entry = {
                "row": tag,
                "reason": d["change_gate_reason"],
                "n_changing": d["n_changing"],
                "change_fidelity": d["change_fidelity"],
                "correct_changed_cells": d["correct_changed_cells"],
            }
            if (
                not d["change_gate_pass"]
                and d["change_gate_reason"] != "no_changing_transitions"
                and (d["n_changing"] or 0) > 0
            ):
                origin_discriminating.append(entry)
            else:
                origin_vacuous.append(entry)

    # ---- GATE 2: closes the measured hole -----------------------------------------
    # Rows where the INCUMBENT admits the spurious writer. The repair must reject a strict
    # superset-by-row of those, or it has not closed anything.
    hole_rows, hole_closed, hole_open = [], [], []
    for r in ok:
        for mask_on in (0, 1):
            d = r["decisions"].get(f"mask={mask_on}|engine=base_plus_spurious")
            if d is None:
                continue
            tag = f"{r['game']}@{r['seed']}|mask={mask_on}"
            if d["incumbent_trust_pass"]:
                hole_rows.append(tag)
                (hole_closed if not d["change_gate_pass"] else hole_open).append(tag)

    # ---- THE RESIDUAL, CHARACTERIZED (not merely counted) -------------------------
    # A residual reported as "62 of 68" is a number, not an explanation, and an unexplained
    # residual is indistinguishable from a bug. Every row that still admits the spurious
    # writer is required to carry the COMPUTED WITNESS of WHY: the no-op channel -- which is
    # what actually catches the spurious writer on the rows that ARE closed -- cannot fire
    # when the corpus contains no true no-op, and union fidelity alone is an ABSOLUTE
    # threshold that a single invented cell moves only slightly when the true change set is
    # large. So the prediction is exact: the escaping set is precisely the rows where
    # `noop_channel_measurable` is False. If a row escapes for any OTHER reason, this
    # characterisation is wrong and the gate below fails rather than papering over it.
    residual_witness = []
    residual_unexplained = []
    for r in ok:
        for mask_on in (0, 1):
            d = r["decisions"].get(f"mask={mask_on}|engine=base_plus_spurious")
            if d is None or not d["incumbent_trust_pass"] or not d["change_gate_pass"]:
                continue
            tag = f"{r['game']}@{r['seed']}|mask={mask_on}"
            w = {
                "row": tag,
                "n_noop": d["n_noop"],
                "noop_channel_measurable": d["noop_channel_measurable"],
                "noop_ok_is_vacuous": d["noop_ok_is_vacuous"],
                "change_fidelity": d["change_fidelity"],
                "n_changing": d["n_changing"],
                "invented_changed_cells": d["invented_changed_cells"],
                "invented_change_rate": d["invented_change_rate"],
            }
            residual_witness.append(w)
            if d["noop_channel_measurable"]:
                residual_unexplained.append(w)

    # Does the PURE invented-write quantity separate on exactly the rows the shipped gate
    # cannot? Measured, published, and deliberately NOT turned into a threshold here.
    # PAIRED per row, and rows where the quantity is structurally unmeasurable are EXCLUDED
    # rather than scored as 0.0. `invented_change_rate` divides by `n_changing`, so a corpus
    # whose held-out split contains no changing transition (ft09: 0 of 40) yields 0.0 for
    # EVERY engine -- the value meaning "invents nothing" doubling as the value meaning "not
    # measurable", the same defect this change flags for the no-op channel. Pooling those
    # rows into a min/max drags the attack minimum to 0.0 and makes a real separation look
    # like no separation. That is not a hypothetical: it is what the first run of this
    # experiment reported.
    invented_sep, invented_excluded = [], []
    for r in ok:
        for mask_on in (0, 1):
            b = r["decisions"].get(f"mask={mask_on}|engine=base")
            a = r["decisions"].get(f"mask={mask_on}|engine=base_plus_spurious")
            if b is None or a is None:
                continue
            entry = {
                "row": f"{r['game']}@{r['seed']}|mask={mask_on}",
                "n_changing": b["n_changing"],
                "honest_invented_rate": b["invented_change_rate"],
                "attack_invented_rate": a["invented_change_rate"],
            }
            if not (b["n_changing"] or 0) > 0:
                entry["excluded_reason"] = "no_changing_transitions_quantity_undefined"
                invented_excluded.append(entry)
                continue
            entry["attack_exceeds_honest"] = (a["invented_change_rate"] or 0.0) > (
                b["invented_change_rate"] or 0.0
            )
            invented_sep.append(entry)
    honest_max = (
        max((x["honest_invented_rate"] or 0.0) for x in invented_sep) if invented_sep else None
    )
    attack_min = (
        min((x["attack_invented_rate"] or 0.0) for x in invented_sep) if invented_sep else None
    )
    n_paired_attack_higher = sum(1 for x in invented_sep if x["attack_exceeds_honest"])

    # ---- GATE 3: MUST-NOT-FIRE control --------------------------------------------
    hw = [r for r in ok if r["game"] == HANDWRITTEN_GAME]
    hw_admitted_mask_on = [
        f"{r['game']}@{r['seed']}"
        for r in hw
        if (r["decisions"].get("mask=1|engine=base") or {}).get("change_gate_pass")
    ]
    hw_admitted_mask_off = [
        f"{r['game']}@{r['seed']}"
        for r in hw
        if (r["decisions"].get("mask=0|engine=base") or {}).get("change_gate_pass")
    ]
    # ---- CORRIGENDUM 2026-07-27 (adversarial review, two defects) --------------------
    # DEFECT A -- THE MASK ARM WAS A SILENT NO-OP. This harness passed `hud_mask=<a real
    # mask>` to `select_trusted_world_model` but never set CARNOT_ARC_WM_HUD_MASK, and
    # `WorldModelVerifier` discards an unflagged mask. Every `mask=1` row in the ORIGINAL
    # run recorded `hud_mask_status="disabled"`, and `change_fidelity` differed between the
    # mask=0 and mask=1 arms on 0 of 162 paired arms -- the experiment measured mask-off
    # TWICE and reported it as "both mask settings". (`incumbent_consistency` DID differ on
    # 9 of 162, because `score_change_weighted_consistency` masked unconditionally: two
    # comparators, two conventions, inside one decision.) Fixed in the library by
    # `arc_world_model_trust_energy.resolve_hud_mask_enabled`; asserted HERE so the harness
    # can never again report an arm it did not run.
    #
    # DEFECT B -- THE ACCEPTANCE GATE COULD NOT FAIL. It read
    # `len(hw_admitted_mask_on) > 0`: it looked ONLY at the mask-ON population (the one
    # condition where the control cannot fail) and passed on ANY ONE of three seeds (an
    # any-seed union, which this project's measurement discipline forbids in favour of
    # per-seed matching). Measured directly through the production path on the WHOLE dc22
    # corpus, the hand-written correct engine scores change_fidelity 0.4694 / 0.4083 /
    # 0.4103 with the mask OFF -- BELOW the 0.5 threshold, REJECTED 3/3 -- and 0.8148 /
    # 0.7609 / 0.7358 with it ON, admitted 3/3.
    #
    # THE CONSEQUENCE, which is the operationally important part: THE GATE MUST NOT SHIP
    # WITHOUT THE MASK. The gate-only arm (A2) rejects the one engine known to be genuinely
    # good. That is not a tuning detail -- it means the two flags are NOT independent in the
    # direction that matters for a flip decision, and A2 is a strictly-worse arm rather than
    # a partial improvement.
    hw_seeds = sorted({r["seed"] for r in hw})
    hw_mask_status_mask_on = sorted(
        {str((r["decisions"].get("mask=1|engine=base") or {}).get("hud_mask_status")) for r in hw}
    )
    # The mask arm is only meaningful if masking actually happened somewhere in it.
    mask_arm_effective = any(
        str((r["decisions"].get("mask=1|engine=base") or {}).get("hud_mask_status"))
        in ("applied", "refused_swallows_dynamics")
        for r in ok
    )
    hw_admitted_both_conditions_every_seed = bool(hw_seeds) and all(
        (r["decisions"].get("mask=1|engine=base") or {}).get("change_gate_pass")
        and (r["decisions"].get("mask=0|engine=base") or {}).get("change_gate_pass")
        for r in hw
    )
    hw_admitted_mask_on_every_seed = bool(hw_seeds) and all(
        (r["decisions"].get("mask=1|engine=base") or {}).get("change_gate_pass") for r in hw
    )

    # ---- GATE 4: identity rejected everywhere -------------------------------------
    identity_admitted = _sel("identity", 1, "change_gate_pass", lambda v: bool(v)) + _sel(
        "identity", 0, "change_gate_pass", lambda v: bool(v)
    )

    # ---- GATE 5: the record is populated in every arm (no dead channel) -----------
    unpopulated = [
        f"{r['game']}@{r['seed']}|{k}"
        for r in ok
        for k, d in r["decisions"].items()
        if not d.get("change_gate_record_populated")
    ]

    # ---- GATE 6: the two verdicts genuinely DISAGREE somewhere --------------------
    # If they never disagreed, the repair would be a no-op dressed as a change, and every
    # other gate above would be passing for the wrong reason.
    disagreements = []
    for r in ok:
        for k, d in r["decisions"].items():
            if bool(d["incumbent_trust_pass"]) != bool(d["change_gate_pass"]):
                disagreements.append(
                    {
                        "row": f"{r['game']}@{r['seed']}|{k}",
                        "incumbent_trust_pass": d["incumbent_trust_pass"],
                        "change_gate_pass": d["change_gate_pass"],
                        "reason": d["change_gate_reason"],
                    }
                )

    gates = {
        "acceptance_gate_new_gate_rejects_real_ondisk_degenerates": (
            len(origin_rows) > 0 and not origin_admitted
        ),
        # The one that actually carries the origin-incident claim. Rejecting an engine on a
        # corpus where NOTHING could pass is not a catch.
        "acceptance_gate_origin_rejection_is_discriminating_not_vacuous": (
            len(origin_discriminating) > 0
        ),
        # The repair must close MOST of the hole and EXPLAIN the rest. Requiring "all" here
        # would be dishonest in the other direction: it would push toward fitting a
        # threshold to the six escaping rows, which is precisely the over-fit this
        # experiment declines to commit. What is required instead is that every escape is
        # accounted for by the named structural mechanism.
        "acceptance_gate_hole_majority_closed": (
            len(hole_rows) > 0 and len(hole_closed) > len(hole_open)
        ),
        "acceptance_gate_every_residual_escape_is_structurally_explained": (
            len(residual_witness) > 0 and not residual_unexplained
        ),
        # PER-SEED, mask-ON. Replaces the any-seed `len(...) > 0` union. Named for the
        # condition it actually tests, so it cannot be read as an unconditional claim.
        "acceptance_gate_admits_handwritten_correct_engine_mask_on_every_seed": (
            hw_admitted_mask_on_every_seed
        ),
        # The MASK ARM RAN AT ALL. Without this the two gates above are satisfiable by an
        # experiment that never enabled masking -- which is exactly what the original run
        # did on all 162 arms.
        "acceptance_gate_mask_arm_actually_masked": bool(mask_arm_effective),
        "acceptance_gate_identity_rejected_on_every_row": (len(ok) > 0 and not identity_admitted),
        "acceptance_gate_change_gate_record_populated_on_every_arm": (
            len(ok) > 0 and not unpopulated
        ),
        "acceptance_gate_two_verdicts_actually_disagree": len(disagreements) > 0,
    }
    # A gate decided over an empty population is not a pass. Checked explicitly rather than
    # folded into the booleans above, so the artifact SHOWS the population sizes.
    populations = {
        "origin_incident_rows": len(origin_rows),
        "rows_where_incumbent_admits_spurious_writer": len(hole_rows),
        "residual_escape_rows": len(residual_witness),
        "handwritten_control_rows": len(hw),
        "total_ok_rows": len(ok),
        "total_decision_arms": sum(len(r["decisions"]) for r in ok),
        "disagreement_rows": len(disagreements),
    }
    gates["acceptance_gate_no_gate_decided_over_empty_population"] = all(
        v > 0 for v in populations.values()
    )
    gates["acceptance_gate_passed"] = all(gates.values())

    analyser_wall = time.time() - analyser_t0
    # THE MEASUREMENT CLOCK IS NOT THE ANALYSER CLOCK. `measurement_wall_s` sums each row's
    # own elapsed_s (the real per-row measurement cost); `duration_s` is this process's
    # wall time. They are different quantities and are reported separately -- reporting one
    # as the other is how an artifact ends up claiming a measurement it did not make.
    measurement_wall = round(sum(float(r.get("elapsed_s", 0.0)) for r in rows), 3)

    artifact = {
        "experiment": "experiment_6013_hidden_state_change_gate_closure",
        "experiment_id": 6013,
        "title": "REQ-ARC-WMTE-6013: close the change gate's hidden-state coverage hole",
        "requirement": "REQ-ARC-WMTE-6013",
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "substrate_note": (
            "No GPU, no model, no LLM. Pure numpy over cached offline-arcade transitions "
            "from e3.collect_transitions, scored through the production "
            "select_trusted_world_model. GGUF names appear nowhere because none is invoked."
        ),
        # Truthfully declares that NO model was invoked, matching the sibling exp6011/6012
        # convention. A null here reads to adversarial_verify as unverifiable methodology,
        # and naming a GGUF that was never loaded would be a vestigial false claim of live
        # inference -- the exact failure the Inference-Substrate discipline exists to stop.
        "model_specs": {
            "note": "no model invoked; pure numpy over cached offline-arcade transitions",
            "engines_scored": (
                "real on-disk results/arc_e3/<game>/world_model.py via e3.load_engine, "
                "plus constructed identity / spurious-writer / hand-written control arms"
            ),
        },
        "random_seed": int(args.seeds[0]),
        "random_seeds_used": [int(s) for s in args.seeds],
        "preconditions_checked": preconditions,
        "duration_s": round(analyser_wall, 3),
        "measurement_wall_s": measurement_wall,
        "n_rows": len(rows),
        "rows": rows,
        "populations": populations,
        "FINDING_origin_incident_ondisk_engines_rejected": sorted(origin_rejected),
        "FINDING_origin_incident_ondisk_engines_admitted": sorted(origin_admitted),
        "FINDING_origin_incident_rows_the_incumbent_would_have_admitted": sorted(
            origin_incumbent_admitted
        ),
        "FINDING_origin_rejections_that_are_DISCRIMINATING": origin_discriminating,
        "FINDING_origin_rejections_that_are_VACUOUS_and_prove_nothing": origin_vacuous,
        "FINDING_rows_where_incumbent_admits_spurious_writer": sorted(hole_rows),
        "FINDING_of_those_rows_now_rejected_by_the_repair": sorted(hole_closed),
        "FINDING_of_those_rows_still_admitted": sorted(hole_open),
        "FINDING_residual_escape_witness": residual_witness,
        "FINDING_residual_escapes_NOT_explained_by_dead_noop_channel": residual_unexplained,
        "FINDING_pure_invented_write_separation": {
            "note": (
                "The purer quantity (cells written where reality changed NOTHING) as "
                "distinct from spurious_changed_cells (which also counts wrong-valued "
                "predictions at genuinely changed cells). Published as calibration input "
                "for a follow-up decision; NOT a gate condition in this change."
            ),
            "n_measurable_paired_rows": len(invented_sep),
            "n_rows_excluded_as_unmeasurable": len(invented_excluded),
            "rows_excluded_as_unmeasurable": invented_excluded,
            "max_over_honest_engines": honest_max,
            "min_over_spurious_writers": attack_min,
            "separates_cleanly": (
                honest_max is not None and attack_min is not None and honest_max < attack_min
            ),
            "n_paired_rows_where_attack_exceeds_honest": n_paired_attack_higher,
            "paired_separation_is_total": (
                len(invented_sep) > 0 and n_paired_attack_higher == len(invented_sep)
            ),
            "why_not_shipped_as_a_gate": (
                "The separation is measured against an engine CONSTRUCTED to invent a cell "
                "on every transition. That says nothing about where a realistically "
                "imperfect induced engine sits, and a threshold fitted to this gap would "
                "reject one. Recalibration against an imperfect engine is the prerequisite."
            ),
            "per_row": invented_sep,
        },
        "FINDING_handwritten_control_admitted_mask_on": sorted(hw_admitted_mask_on),
        "FINDING_handwritten_control_admitted_mask_off": sorted(hw_admitted_mask_off),
        # ---- CORRIGENDUM FINDINGS (2026-07-27) --------------------------------------
        # Reported as FINDINGS, deliberately NOT as acceptance gates. The honest expected
        # value of `..._admitted_in_BOTH_conditions_every_seed` is FALSE (the control is
        # rejected mask-off), and a gate whose expected value is failure would either block
        # the artifact forever or invite tuning the threshold until it passes. The finding
        # records the fact; the operator decides what it means for a flip.
        "FINDING_handwritten_control_admitted_in_BOTH_conditions_every_seed": bool(
            hw_admitted_both_conditions_every_seed
        ),
        "FINDING_handwritten_control_seeds": hw_seeds,
        "FINDING_mask_on_arm_hud_mask_status_values": hw_mask_status_mask_on,
        "FINDING_gate_must_not_ship_without_mask": (
            "Measured through the production path on the WHOLE dc22 corpus, the hand-written "
            "correct engine scores change_fidelity 0.4694/0.4083/0.4103 with the mask OFF "
            "(REJECTED 3/3, reason change_fidelity_below_threshold) and 0.8148/0.7609/0.7358 "
            "with it ON (admitted 3/3). The gate-only arm therefore rejects the one engine "
            "known to be genuinely good. Do not flip CARNOT_ARC_WM_CHANGE_GATE without also "
            "flipping CARNOT_ARC_WM_HUD_MASK."
        ),
        "FINDING_original_run_mask_arm_was_a_silent_noop": (
            "The ORIGINAL exp6013 run passed a real mask but never set "
            "CARNOT_ARC_WM_HUD_MASK, so every mask=1 row recorded hud_mask_status='disabled' "
            "and change_fidelity differed between mask=0 and mask=1 on 0 of 162 paired arms. "
            "acceptance_gate_mask_arm_actually_masked now makes that state unreportable."
        ),
        "FINDING_identity_engine_admitted_anywhere": sorted(identity_admitted),
        "FINDING_verdict_disagreements": disagreements[:60],
        "FINDING_arms_with_unpopulated_change_gate_record": unpopulated,
        "interpretation": {
            "what_this_does_not_show": (
                "This is an OFFLINE decision-level measurement. It shows the repaired gate "
                "admits and rejects the engines it should on real cached transitions. It does "
                "NOT show a live win-rate effect -- no episode was played, no plan executed, "
                "no level solved. The live four-arm A/B is the separate measurement that can "
                "speak to that, and nothing here should be read as a substitute for it."
            ),
            "mask_and_gate_are_not_independent_on_the_control": (
                "The hand-written control's admission is mask-sensitive. On the WHOLE dc22 "
                "corpus the mask-off gate REJECTS it on every seed (3/3) and the mask-on gate "
                "ADMITS it on every seed (3/3). Shipping the gate WITHOUT the mask therefore "
                "rejects the one engine known to be genuinely good. The earlier wording of "
                "this field ('Where mask-off rejects it...') hedged a fact that is not "
                "conditional, and the artifact it was written against had a mask arm that "
                "never masked; both are corrected here."
            ),
            "held_out_split_vs_whole_corpus": (
                "This experiment decides on the HELD-OUT split that "
                "select_trusted_world_model owns (n_changing 32/28/28 on dc22), while the "
                "direct whole-corpus measurement quoted above uses all 120 transitions "
                "(n_changing 89/80/91 mask-off). The two populations differ by ~3x and the "
                "control sits within 0.035 of the 0.5 threshold on the split, so its verdict "
                "there is not robust. Whole-corpus is the figure to quote for a flip decision; "
                "the split figure is what the LIVE code path actually computes. Both are "
                "reported rather than reconciled away."
            ),
            "flag_state": (
                "Nothing was flipped. SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED=False, "
                "SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED=False, "
                "SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED=None (follow). The "
                "arms above are selected by passing hud_mask explicitly and by reading the "
                "gate's computed record, never by mutating a shipped constant."
            ),
        },
        "verifier_is_oracle": True,
        "verifier_is_oracle_note": (
            "TRUE and deliberately so. The engines here are graded against the recorded "
            "true next grids, which ARE the oracle for dynamics prediction. This is a "
            "measurement of a GATE's discrimination, not a moat claim, and none of these "
            "numbers is headline- or gate-flip-eligible on their own."
        ),
        **gates,
        "provenance": _provenance(),
        "honest_verdict": (
            "complete_hidden_state_change_gate_closed_and_verified"
            if gates["acceptance_gate_passed"]
            else "complete_hidden_state_change_gate_measured_with_failing_gate"
        ),
    }
    artifact["reproducibility_checksum"] = _sha(
        {
            "rows": [{k: v for k, v in r.items() if k != "elapsed_s"} for r in rows],
            "seeds": args.seeds,
            "n": args.n,
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=2, default=str))
    print(
        json.dumps({k: v for k, v in artifact.items() if k.startswith("acceptance_gate")}, indent=2)
    )
    print("populations:", json.dumps(populations))
    print("wrote", OUT)
    return 0 if gates["acceptance_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

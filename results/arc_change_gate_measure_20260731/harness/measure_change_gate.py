"""PHASE 4 -- measure the change-aware trust gate on BOTH sides.

THE QUESTION, stated so it cannot be answered by only its flattering half:

  (a) Does the change gate CATCH the vacuous engines the 2026-07-30 force-admit audit
      named -- lp85's engine that invents 96 cells while getting 0 of 2 real changes
      right, and ft09-onb which hallucinates a change on 19 of 19 no-ops?
  (b) Does it still PASS the one structurally sound engine in that run -- tn36-on,
      8/8 held-out exact, cell_recall 1.0, 0 invented cells?

A gate that catches junk by ALSO killing the one good model is worse than no gate. So
both halves are measured, on the SAME engines, in the SAME script, and both are reported
whatever they say. Nothing here flips a shipped default: this writes an artifact.

WHAT IS BEING GRADED, and how we know it is the right thing.

The engines are the BYTE-EXACT sources from the force-admit audit
(results/outer_loop_arc_gate_forceadmit_20260730.json), recovered from the retention
store by sha256 -- every staged file's sha256[:16] equals that cell's recorded
`src_engine_sha256`. tn36-on additionally equals the frozen Phase-3 fixture
(tests/fixtures/arc_goal_gate_depth_tn36/tn36_on_world_model.py.frozen, md5
6d96491f80bec0319828ba1a04f5841e), so the good-engine control is the same object Phase 3
analysed.

The transitions are the SECOND induce call's corpus captured in
results/arc_engine_validation_20260731/harness/capture/<game>/transitions2.pkl, at the
same seed (1) and budget (60) as the audit cells. That corpus is not asserted to be the
audit's -- it is CHECKED to be, per row, by reproducing the audit's own recorded
`n_transitions` / `n_changing` / `n_noop` / `verify_accuracy` / `verify_cell_recall` /
`invented_changed_cells` / `n_noop_hallucinated`. Every reproduction is reported
field-by-field in `fidelity_vs_audit`, agreements AND disagreements, because a
measurement that silently grades a different corpus than the one it claims is exactly the
failure this project keeps paying for. A row whose fidelity check fails is reported and
EXCLUDED from the verdict rather than quietly averaged in.

WHAT IS DELIBERATELY *NOT* DONE HERE.

No default is changed. No threshold is lowered. `change_gate_decision` is called with
`enabled=True` as an ARGUMENT -- the shipped constant is untouched, so this script cannot
alter live behaviour even if it is re-run by accident.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))

SCRATCH = Path(
    "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/p4gate/engines"
)
CAPTURE = REPO / "results/arc_engine_validation_20260731/harness/capture"
AUDIT = REPO / "results/outer_loop_arc_gate_forceadmit_20260730.json"
OUT = REPO / "results/arc_change_gate_measure_20260731/change_gate_two_sided.json"

# The nine cells the audit produced an engine for. sc25 has only an `on` arm (its `onb`
# terminal label is a GENERATION failure, not a gate rejection -- see the audit).
CELLS = [
    ("lp85", "on"),
    ("lp85", "onb"),
    ("ft09", "on"),
    ("ft09", "onb"),
    ("tn36", "on"),
    ("tn36", "onb"),
    ("tu93", "on"),
    ("tu93", "onb"),
    ("sc25", "on"),
]

# The two engines the diagnosis names as the vacuous cases the gate MUST catch, and the
# one it MUST NOT kill. Named explicitly so the verdict is computed against a
# pre-registered target list rather than chosen after seeing the numbers.
MUST_CATCH = {("lp85", "on"), ("ft09", "onb")}
MUST_NOT_FIRE = {("tn36", "on")}


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_engine_from(path: Path) -> tuple[Any, Any, str | None]:
    """Load (engine, is_level_complete) from a standalone world_model.py source file.

    Mirrors e3._load_engine_from, but takes a FILE rather than a (root, game) pair,
    because the staged engines are flat files keyed by cell, not a store layout.
    """
    spec = importlib.util.spec_from_file_location(f"wm_{path.stem}", path)
    if spec is None or spec.loader is None:
        return None, None, "spec_failed"
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except Exception as exc:  # a syntactically broken engine is a real, reportable state
        return None, None, f"{type(exc).__name__}: {exc}"[:200]
    return getattr(mod, "engine", None), getattr(mod, "is_level_complete", None), None


def audit_record(audit: dict, game: str, arm: str) -> dict:
    return audit["live_gate_records_per_cell"].get(f"{game}___{arm}", {})


def main() -> int:
    t0 = time.time()
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_llm_reinduction import _proposal_prefix

    audit = json.loads(AUDIT.read_text())
    rows: list[dict[str, Any]] = []

    for game, arm in CELLS:
        src = SCRATCH / f"{game}__{arm}.py"
        rec = audit_record(audit, game, arm)
        row: dict[str, Any] = {
            "game": game,
            "arm": arm,
            "engine_sha256_16": sha16(src) if src.exists() else None,
            "audit_src_engine_sha256": None,
            "engine_source_verified": False,
        }
        # The audit stores the engine hash on the force-admit cells, not the gate record.
        for cell in audit["force_admit_cells"]:
            if cell["game"] == game and cell["engine_arm"] == arm and cell.get("src_engine_sha256"):
                row["audit_src_engine_sha256"] = cell["src_engine_sha256"]
                break
        row["engine_source_verified"] = (
            row["engine_sha256_16"] is not None
            and row["engine_sha256_16"] == row["audit_src_engine_sha256"]
        )

        tpath = CAPTURE / game / "transitions2.pkl"
        if not tpath.exists():
            row["status"] = "no_transitions"
            rows.append(row)
            continue
        with tpath.open("rb") as fh:
            transitions = pickle.load(fh)

        engine, _is_done, err = load_engine_from(src)
        if engine is None:
            row["status"] = f"unloadable:{err}"
            rows.append(row)
            continue

        # --- the shipped verifier, mask OFF (the audit's own hud_mask_status) ----------
        vr = e3.WorldModelVerifier(list(transitions), hud_mask=None).score(engine)

        # --- fidelity vs the audit's own recorded numbers -----------------------------
        checks = {
            "n_transitions": (int(vr.n), rec.get("n_transitions")),
            "n_changing": (int(vr.n_changing), rec.get("n_changing")),
            "n_noop": (int(vr.n_noop), rec.get("n_noop")),
            "n_noop_hallucinated": (int(vr.n_noop_hallucinated), rec.get("n_noop_hallucinated")),
            "n_changes_correct": (int(vr.n_changes_correct), rec.get("n_changes_correct")),
            "invented_changed_cells": (
                int(vr.invented_changed_cells),
                rec.get("invented_changed_cells"),
            ),
            "verify_accuracy": (round(float(vr.accuracy), 4), rec.get("verify_accuracy")),
            "verify_cell_recall": (round(float(vr.cell_recall), 4), rec.get("verify_cell_recall")),
        }
        fidelity = {
            k: {"measured": m, "audit": a, "agrees": (a is not None and m == a)}
            for k, (m, a) in checks.items()
        }
        n_cmp = sum(1 for v in fidelity.values() if v["audit"] is not None)
        n_ok = sum(1 for v in fidelity.values() if v["agrees"])
        row["fidelity_vs_audit"] = fidelity
        row["fidelity_fields_compared"] = n_cmp
        row["fidelity_fields_agreeing"] = n_ok
        row["fidelity_ok"] = bool(n_cmp > 0 and n_ok == n_cmp)
        if n_cmp == 0:
            # Not a silent zero. sc25-on's audit record carries every one of these keys with
            # value None because its terminal induce ended `proposer_failed_or_missing_root`
            # -- the plain path never ran, so there is nothing to reproduce and no way to
            # confirm this corpus is the one that cell graded. Excluded from the verdict with
            # the reason attached, rather than counted as a silent agreement.
            row["fidelity_excluded_reason"] = (
                "audit record has no populated comparison fields "
                f"(terminal_skipped={rec.get('terminal_skipped')!r}); corpus identity "
                "unconfirmable, row reported but excluded from the verdict"
            )

        # --- the change gate, both arms, on the WHOLE corpus --------------------------
        gate_on = e3.change_gate_decision(vr, enabled=True)
        gate_off = e3.change_gate_decision(vr, enabled=False)

        # --- the incumbent decisions at the SAME site ---------------------------------
        # arc_competition_agent.py:6095 is an if/elif: turning the change gate on REPLACES
        # the `_gate_value < 0.5` accuracy check rather than adding to it. Both incumbent
        # readings are recorded so the substitution can be judged, not assumed.
        row["incumbent_cell_recall_at_0.5"] = bool(float(vr.cell_recall) >= 0.5)
        row["incumbent_accuracy_at_0.5"] = bool(float(vr.accuracy) >= 0.5)
        row["incumbent_accuracy_at_1.0"] = bool(float(vr.accuracy) >= 1.0)

        # --- the OTHER call site: the reinduction path's held-out suffix ---------------
        # execute_bounded_llm_reinduction admits on `heldout_accuracy >= 1.0` over the
        # suffix _proposal_prefix keeps out of the prompt. The change gate is NOT wired
        # there. Scored anyway, so the report can say what WOULD happen if it were.
        prefix = _proposal_prefix(list(transitions))
        heldout = list(transitions)[len(prefix) :]
        row["n_heldout"] = len(heldout)
        if heldout:
            hvr = e3.WorldModelVerifier(heldout, hud_mask=None).score(engine)
            hgate = e3.change_gate_decision(hvr, enabled=True)
            row["heldout"] = {
                "accuracy": round(float(hvr.accuracy), 6),
                "cell_recall": round(float(hvr.cell_recall), 6),
                "n_changing": int(hvr.n_changing),
                "n_noop": int(hvr.n_noop),
                "n_noop_hallucinated": int(hvr.n_noop_hallucinated),
                "correct_changed_cells": int(hvr.correct_changed_cells),
                "invented_changed_cells": int(hvr.invented_changed_cells),
                "change_fidelity": round(float(hvr.change_fidelity), 6),
                "incumbent_admits_at_1.0": bool(float(hvr.accuracy) >= 1.0),
                "change_gate_passed": bool(hgate["passed"]),
                "change_gate_reason": hgate["reason"],
            }

        row["status"] = "ok"
        row["measured"] = {
            "accuracy": round(float(vr.accuracy), 6),
            "cell_recall": round(float(vr.cell_recall), 6),
            "change_fidelity": round(float(vr.change_fidelity), 6),
            "change_accuracy": round(float(vr.change_accuracy), 6),
            "correct_changed_cells": int(vr.correct_changed_cells),
            "spurious_changed_cells": int(vr.spurious_changed_cells),
            "invented_changed_cells": int(vr.invented_changed_cells),
            "invented_change_rate": round(float(vr.invented_change_rate), 6),
            "n_changing": int(vr.n_changing),
            "n_changes_correct": int(vr.n_changes_correct),
            "n_noop": int(vr.n_noop),
            "n_noop_hallucinated": int(vr.n_noop_hallucinated),
            "noop_hallucination_rate": round(float(vr.noop_hallucination_rate), 6),
            "noop_channel_measurable": bool(vr.noop_channel_measurable),
        }
        row["change_gate_on"] = {"passed": bool(gate_on["passed"]), "reason": gate_on["reason"]}
        row["change_gate_off"] = {"passed": bool(gate_off["passed"]), "reason": gate_off["reason"]}

        # ---- DISCRIMINATING vs VACUOUS, per exp6013's own standard --------------------
        # experiment_6013 splits its origin-incident rejections into
        # FINDING_origin_rejections_that_are_DISCRIMINATING and
        # ..._that_are_VACUOUS_and_prove_nothing, on exactly this test: a rejection whose
        # reason is `no_changing_transitions` was decided over an EMPTY population and is
        # evidence of nothing about the engine. Applying the same split here is what stops
        # this measurement from counting its own vacuous rejections as catches.
        #
        # `decision_population` is reported alongside because a non-vacuous rejection over
        # n_changing=2 is still a THIN one, and a reader who sees only "REJECT
        # (change_fidelity_below_threshold)" cannot tell 2 from 40.
        row["decision_population_n_changing"] = int(vr.n_changing)
        row["decision_is_vacuous"] = bool(
            gate_on["reason"] == "no_changing_transitions" or int(vr.n_changing) == 0
        )
        row["decision_is_discriminating"] = bool(
            (not gate_on["passed"]) and not row["decision_is_vacuous"]
        )
        # The residual-escape channel exp6013 found on re86: when n_noop == 0 the no-op
        # test passes because nothing could be tested, not because the engine is clean.
        row["noop_verdict_is_vacuous"] = bool(int(vr.n_noop) == 0)
        row["audit_terminal_skipped"] = rec.get("terminal_skipped")
        row["audit_refinement_round_heldout"] = rec.get("refinement_round_heldout")
        row["audit_refinement_round_skips"] = rec.get("refinement_round_skips")
        rows.append(row)

    # ---- verdict, computed against the PRE-REGISTERED target lists -------------------
    usable = [r for r in rows if r.get("status") == "ok" and r.get("fidelity_ok")]
    caught = {
        (r["game"], r["arm"]): (not r["change_gate_on"]["passed"], r["change_gate_on"]["reason"])
        for r in usable
    }
    by_cell = {(r["game"], r["arm"]): r for r in usable}
    must_catch_result = {}
    for g, a in sorted(MUST_CATCH):
        r = by_cell.get((g, a))
        if r is None:
            must_catch_result[f"{g}__{a}"] = {"measured": None, "reason": "not_measured"}
            continue
        # A VACUOUS rejection does NOT count as a catch. Counting it would let a corpus
        # with no changing transitions masquerade as gate discrimination -- the exact
        # error exp6013 named and split out.
        must_catch_result[f"{g}__{a}"] = {
            "measured": bool(r["decision_is_discriminating"]),
            "rejected_at_all": bool(not r["change_gate_on"]["passed"]),
            "reason": r["change_gate_on"]["reason"],
            "decision_population_n_changing": r["decision_population_n_changing"],
            "vacuous": r["decision_is_vacuous"],
        }
    # The no-op vacuity flag is surfaced INSIDE the must-not-fire verdict, not only in the row
    # (2026-07-31 adversarial review). The gate has three sub-conditions and one of them --
    # "does the engine hallucinate a change on a transition that did not change?" -- is
    # UNTESTABLE when the engine's transition set contains no no-ops at all. tn36-on is exactly
    # that case (`n_noop == 0`), so its pass exercises two of three conditions. The module's own
    # docstring warns that a consumer treating such a verdict as a clean pass "will read a false
    # pass"; presenting `passes_change_gate: True` with no qualifier alongside it is precisely
    # that consumer. Carrying the flag here means any summary quoting the verdict inherits it.
    _row_by_cell = {(r["game"], r["arm"]): r for r in rows}
    must_not_fire_result = {}
    for g, a in sorted(MUST_NOT_FIRE):
        row = _row_by_cell.get((g, a), {})
        meas = row.get("measured") or {}
        vacuous_noop = bool(row.get("noop_verdict_is_vacuous"))
        must_not_fire_result[f"{g}__{a}"] = {
            "passes_change_gate": (None if (g, a) not in caught else (not caught[(g, a)][0])),
            "reason": caught.get((g, a), (None, "not_measured"))[1],
            "n_noop": meas.get("n_noop"),
            "noop_channel_measurable": meas.get("noop_channel_measurable"),
            "noop_verdict_is_vacuous": vacuous_noop,
            "pass_is_qualified": vacuous_noop,
            "qualification": (
                "PASS IS QUALIFIED: n_noop == 0, so the no-op hallucination sub-condition was "
                "not exercised at all. Two of the gate's three sub-conditions were tested on "
                "this engine. Do not cite this as unqualified evidence that the gate spares "
                "good engines."
                if vacuous_noop
                else "All three sub-conditions were exercised on a non-empty population."
            ),
        }
    catches_all_targets = all(v["measured"] is True for v in must_catch_result.values())
    spares_good = all(v["passes_change_gate"] is True for v in must_not_fire_result.values())

    # ---- the admission DELTA vs each incumbent configuration ------------------------
    # arc_competition_agent.py:6078-6100 is an if/elif: enabling the flag REPLACES the
    # incumbent `_gate_value < 0.5` check rather than AND-ing with it. So the honest
    # question is not "does the gate reject bad engines" but "what does the SWAP change,
    # in BOTH directions". A swap that removes false admits but also adds new ones is not
    # a precision win, and only a two-directional count can tell them apart.
    swap: dict[str, list[str]] = {
        "exact_0.5__admit_to_reject": [],
        "exact_0.5__reject_to_admit": [],
        "cell_recall_0.5__admit_to_reject": [],
        "cell_recall_0.5__reject_to_admit": [],
    }
    for r in usable:
        tag = f"{r['game']}-{r['arm']}"
        gate_admits = bool(r["change_gate_on"]["passed"])
        for metric, key in (
            ("incumbent_accuracy_at_0.5", "exact_0.5"),
            ("incumbent_cell_recall_at_0.5", "cell_recall_0.5"),
        ):
            inc = bool(r[metric])
            if inc and not gate_admits:
                swap[f"{key}__admit_to_reject"].append(tag)
            elif gate_admits and not inc:
                swap[f"{key}__reject_to_admit"].append(tag)

    artifact = {
        "experiment": "arc_change_gate_two_sided_measurement",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": (
            "Does the REQ-ARC-WMTE-6011 change-aware trust gate catch the vacuous engines "
            "the 2026-07-30 force-admit audit named (lp85-on: 96 invented cells, 0 of 2 real "
            "changes right; ft09-onb: a hallucinated change on 19 of 19 no-ops) AND still "
            "pass the one structurally sound engine in that run (tn36-on)?"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        # TRUE, and deliberately so -- matching exp6013's declaration for the identical
        # situation. The engines here are graded against the RECORDED TRUE next grids, which
        # ARE the oracle for dynamics prediction. This measures a GATE's discrimination; it
        # is not a moat claim, and no number here is headline- or gate-flip-eligible alone.
        # An earlier draft of this artifact declared False, which was wrong on the
        # Circularity / Oracle-Distinctness Discipline's own definition.
        "verifier_is_oracle": True,
        "verifier_is_oracle_note": (
            "The comparison target is the recorded next_grid, i.e. the executable oracle that "
            "defines dynamics correctness. Circular by construction and reported as such."
        ),
        "solve_provenance": "development_proxy",
        "random_seed": 1,
        "duration_s": round(time.time() - t0, 4),
        "rows": rows,
        "verdict": {
            "must_catch": must_catch_result,
            "must_not_fire": must_not_fire_result,
            "catches_all_named_vacuous_cases": bool(catches_all_targets),
            "spares_the_good_engine": bool(spares_good),
            "both_conditions_hold": bool(catches_all_targets and spares_good),
            "recommendation_rule": (
                "Recommend enabling CARNOT_ARC_WM_CHANGE_GATE only if BOTH hold. Otherwise "
                "report the measurement and leave the shipped default as-is."
            ),
            "recommendation": (
                "NO -- DO NOT ENABLE THE GATE MASK-OFF, and nothing was acted on. This "
                "measurement's own two-sided criterion is met on THIS population (both named "
                "vacuous engines rejected for discriminating reasons, tn36-on passes, and the "
                "admission swap adds ZERO new admits against either incumbent). But the "
                "criterion is a necessary condition, not the decision, and the decision is "
                "settled AGAINST by the priors: this run was measured MASK OFF, and both "
                "exp6011 and exp6013 measured the hand-written CORRECT dc22 control REJECTED "
                "mask-off on 3 of 3 seeds on the whole corpus (change_fidelity 0.4694 / 0.4083 "
                "/ 0.4103), with exp6013's own FINDING_gate_must_not_ship_without_mask stating "
                "verbatim: 'the gate-only arm therefore rejects the one engine known to be "
                "genuinely good. Do not flip CARNOT_ARC_WM_CHANGE_GATE without also flipping "
                "CARNOT_ARC_WM_HUD_MASK.' An earlier draft of this field read QUALIFIED YES and "
                "described that as an unresolved exp6011-vs-exp6013 disagreement; it is not a "
                "disagreement (see scope_limits), and calling it one was this artifact's error. "
                "Three further limits stand regardless: the must-not-fire side is n=1 AND that "
                "one pass is QUALIFIED (tn36-on has n_noop == 0, so one of the gate's three "
                "sub-conditions was never exercised); the flag does not govern the call site "
                "where the lp85 acceptance the diagnosis names actually happened; and the "
                "predicted funnel effect is zero. "
                "SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED is left False, as shipped."
            ),
            "recommendation_short": "NO for mask-off; not measured for mask-on",
            "what_would_settle_it": (
                "Re-run this two-sided measurement MASK ON, since mask-off is already decided "
                "against by the dc22 control. Then widen the must-not-fire population beyond "
                "one: score a set of engines INDEPENDENTLY known to be good (the hand-written "
                "controls exp6011/6013 already build, plus any engine whose plan matched "
                "reality) through the plain branch, and include at least one whose transition "
                "set contains real no-ops so the third sub-condition is actually exercised."
            ),
        },
        "admission_swap_vs_incumbent": swap,
        "n_rows": len(rows),
        "n_usable": len(usable),
        "flag_default_unchanged": True,
        "thresholds_unchanged": True,
        "scope_limits": {
            "branch_coverage": (
                "Every row here is scored the way the PLAIN (non-hidden-state) branch scores. "
                "sc25 is in HIDDEN_STATE_GAME_IDS, so its live admission runs the REQ-6013 "
                "hidden-state branch instead and its row below is NOT its live decision. The "
                "three pre-registered targets (lp85-on, ft09-onb, tn36-on) are all plain-branch."
            ),
            "WHY_sc25_IS_EXCLUDED_TWO_INDEPENDENT_REASONS": (
                "Stated together because a summary quoting only one of them reads as a "
                "contradiction of the other (2026-07-31 adversarial review). (1) FIDELITY: the "
                "sc25-on AUDIT record has NO populated comparison fields -- its terminal induce "
                "ended `proposer_failed_or_missing_root`, so all eight audit-side values are "
                "null and corpus identity cannot be confirmed. That is what sets "
                "`fidelity_ok: false` and drops it from `usable`. It does NOT mean the row has "
                "no numbers: this run's OWN measured side is fully populated (n_changing 7, "
                "n_noop 18, verify_accuracy 0.76, verify_cell_recall 0.2778) and is reported. "
                "(2) BRANCH: sc25 is a hidden-state game, so even a fidelity-confirmed "
                "plain-branch row would not be its live decision. Either reason alone excludes "
                "it from the verdict; both are true."
            ),
            "REJECTED_ENGINE_COUNT_STATED_EXACTLY": (
                "9 rows; 8 usable (sc25-on excluded, above); of those 8 the gate rejects 7 and "
                "passes 1 (tn36-on). Two of the 7 are the pre-registered must-catch targets, so "
                "the count of OTHER rejected engines is 5 -- lp85-onb, ft09-on, tn36-onb, "
                "tu93-on, tu93-onb. A summary saying 'six other engines' is reachable only by "
                "counting the sc25 row that the same summary excludes; the number is 5."
            ),
            "mask_condition": (
                "Measured MASK OFF, matching every audit cell's own hud_mask_status='disabled'. "
                "This population therefore says nothing about the mask-coupling that exp6011 "
                "measured on dc22, and does not overturn it."
            ),
            "known_residual_escape": (
                "exp6013's FINDING_residual_escape_witness: on re86 the no-op channel is DEAD "
                "(n_noop=0) and an engine with invented_change_rate 1.0 clears the gate at "
                "change_fidelity 0.919. `noop_verdict_is_vacuous` is recorded per row here for "
                "the same reason -- a no-op verdict over an empty no-op population is not a pass."
            ),
            "no_live_effect_claimed": (
                "This is an offline decision-level measurement over 9 cached engine/corpus "
                "pairs. No episode was played and no level was banked. It says which engines "
                "the gate admits, NOT that admitting them moves the funnel."
            ),
            "THE_FLAG_DOES_NOT_GOVERN_THE_SITE_THE_lp85_ACCEPTANCE_HAPPENED_AT": (
                "`change_gate_decision` is called ZERO times in arc_llm_reinduction.py "
                "(verified by grep). That module admits a refinement round on "
                "`heldout_accuracy >= verifier_threshold` alone. The diagnosis's phrase "
                "'lp85 rounds 2-3 scored heldout 1.0 and were ACCEPTED' describes THAT "
                "admission -- so flipping CARNOT_ARC_WM_CHANGE_GATE would NOT have changed "
                "those two rounds' fate. The flag decides at exactly one place for these "
                "games: arc_competition_agent.py:6078-6100. Every catch claimed in this "
                "artifact is a catch AT THAT SITE and nowhere else."
            ),
            "WHICH_lp85_ENGINE_THIS_ACTUALLY_MEASURES": (
                "The diagnosis conflates two engines. The rounds ACCEPTED at heldout 1.0 are "
                "rounds 2 and 3; the engine carrying the 96-invented-cell diagnostics is the "
                "RETAINED one (engine_retention.best_round == 1 for lp85-on). This row "
                "measures the retained engine -- confirmed by reproducing the audit's own "
                "verify_accuracy 0.92 / verify_cell_recall 0.4369 / invented_changed_cells 96 "
                "exactly. Rounds 2 and 3's sources were never retained and are NOT "
                "recoverable, so what the gate would have done to THEM is unmeasured and is "
                "not claimed here."
            ),
            "WHY_THE_CATCH_MECHANISM_IS_NOT_THE_ONE_THE_DIAGNOSIS_IMPLIES": (
                "The diagnosis cites lp85's `n_changes_correct: 0`, which is TRANSITION-level "
                "exact match (0 of 2) and is reproduced here. The gate's non-degeneracy floor "
                "is a different, CELL-level quantity, `correct_changed_cells`, which measures "
                "256 -- so the `degenerate_engine_no_correct_changed_cells` branch does NOT "
                "fire on lp85. The catch is `change_fidelity` 0.375 < 0.5, a 0.125 margin over "
                "an n_changing of 2. Real, but thin, and not the mechanism the framing implies."
            ),
            "funnel_effect_predicted_zero": (
                "The funnel metric is 'LLM output reaches the policy', today 1 of 6. Neither "
                "engine the gate newly rejects was reaching the policy: lp85-on ended "
                "`goal_predicate_error` and ft09-onb ended `world_model_accuracy_below_"
                "threshold`, both with levels_gained 0. And the ONE cell that did reach the "
                "policy, vc33, has every change-gate diagnostic field NULL in the audit -- its "
                "engine never arrived at this call site (terminal_skipped "
                "`degenerate_goal_predicate`). So this gate cannot be shown either to preserve "
                "or to break the single funnel success. Predicted funnel effect: 0 of 6 -> 0 "
                "of 6 change. The gate is a PRECISION change, not a funnel change."
            ),
            "n_equals_1_on_the_must_not_fire_side": (
                "'Spares the good engine' rests on ONE engine (tn36-on), because that run "
                "produced only one structurally sound engine. One passing engine cannot "
                "establish a false-reject RATE -- and that single pass is itself QUALIFIED: "
                "tn36-on has n_noop == 0, so the gate's no-op hallucination sub-condition was "
                "never exercised on it (surfaced as `noop_verdict_is_vacuous` / "
                "`pass_is_qualified` inside verdict.must_not_fire, not only in the row)."
            ),
            "THE_PRIORS_DO_NOT_DISAGREE_AND_THEY_SETTLE_MASK_OFF": (
                "CORRECTED 2026-07-31 (adversarial review). An earlier draft of this field "
                "described exp6011 and exp6013 as an unresolved disagreement about the "
                "hand-written correct dc22 control mask-off. They do not disagree on the "
                "measurement that decides a flip. exp6011's `must_not_fire_control` records "
                "whole_corpus_mask_off False on all three seeds (change_fidelity 0.4694 / "
                "0.4083 / 0.4103) with `mask_flips_reject_to_admit_all_seeds: true`. exp6013 "
                "reports the SAME whole-corpus result and states it as a FINDING: "
                "'FINDING_gate_must_not_ship_without_mask ... the gate-only arm therefore "
                "rejects the one engine known to be genuinely good. Do not flip "
                "CARNOT_ARC_WM_CHANGE_GATE without also flipping CARNOT_ARC_WM_HUD_MASK.' What "
                "differs is only exp6013's HELD-OUT-SPLIT row (the population the live "
                "`select_trusted_world_model` actually owns), where the control is admitted "
                "mask-off 3/3 -- and exp6013 resolves that itself: its "
                "`held_out_split_vs_whole_corpus` field notes the split populations differ by "
                "~3x, the control sits within 0.035 of the threshold there so 'its verdict "
                "there is not robust', and 'whole-corpus is the figure to quote for a flip "
                "decision'. Applying this artifact's own recommendation_rule to that, the "
                "honest call for MASK OFF -- which is what this run measured -- is NO. Calling "
                "it a tie was this artifact's error and is corrected rather than removed."
            ),
            "admit_rate_on_the_broader_engine_corpus_is_zero": (
                "exp6011 reports new_gate_admits_ondisk_mask_on == 0 and _mask_off == 0 across "
                "75 rows (25 games x 3 seeds) of real on-disk engines. A gate that admits 1 of "
                "9 here and 0 of 75 there may be closer to 'rejects everything' than to "
                "'rejects the bad ones'. On this population that is the correct behaviour "
                "(8 of the 9 engines ARE broken), but it is the strongest single reason not to "
                "read a 2-catch / 1-spare result as a licence to flip a submission default."
            ),
        },
    }
    body = json.dumps(artifact, indent=1, sort_keys=True)
    artifact["reproducibility_checksum"] = hashlib.sha256(body.encode()).hexdigest()[:32]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n")
    print(f"wrote {OUT}")
    for r in rows:
        if r.get("status") != "ok":
            print(f"  {r['game']:5s} {r['arm']:4s} {r['status']}")
            continue
        m = r["measured"]
        print(
            f"  {r['game']:5s} {r['arm']:4s} fid={r['fidelity_fields_agreeing']}/"
            f"{r['fidelity_fields_compared']} acc={m['accuracy']:.3f} cr={m['cell_recall']:.3f} "
            f"cf={m['change_fidelity']:.3f} ccc={m['correct_changed_cells']:3d} "
            f"inv={m['invented_changed_cells']:3d} noop={m['n_noop_hallucinated']}/{m['n_noop']} "
            f"| gate={'PASS' if r['change_gate_on']['passed'] else 'REJECT'} "
            f"({r['change_gate_on']['reason']})"
        )
    print(json.dumps(artifact["verdict"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

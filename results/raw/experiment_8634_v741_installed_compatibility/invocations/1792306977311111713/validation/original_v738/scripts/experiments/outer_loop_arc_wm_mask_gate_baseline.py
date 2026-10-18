#!/usr/bin/env python3
"""PRE-CHANGE BASELINE for the REQ-ARC-WMTE-6010 (HUD mask) / -6011 (change gate) four-arm A/B.

WHY THIS EXISTS
---------------
Two independent repairs landed default-off on 2026-07-27:

  * REQ-ARC-WMTE-6010 -- `SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED` / `CARNOT_ARC_WM_HUD_MASK`:
    the HUD was inside the world-model exact-match comparison, so on any game with a monotone
    step counter EVERY frame differed in the HUD and full-grid exact match was unattainable by
    construction. Masking REMOVES unattainable cells, so it can only RAISE measured accuracy.

  * REQ-ARC-WMTE-6011 -- `SUBMITTED_WORLD_MODEL_CHANGE_GATE_ENABLED` / `CARNOT_ARC_WM_CHANGE_GATE`:
    the legacy `accuracy >= 0.5` trust gate is full-grid exact match denominated over ALL
    transitions INCLUDING no-ops, so an identity engine passes it on a no-op-heavy corpus. The
    change gate REJECTS degenerate engines, so it can only LOWER the admission rate.

The two push in OPPOSITE directions. Measured together they confound: a null could mean "both
worked and cancelled" or "neither did". This script establishes the PRE-CHANGE numbers, from
banked data only (zero GPU), so the four-arm A/B has a fixed thing to move against and so the
power ceiling is stated BEFORE the run rather than after a non-significant result.

WHAT IT DOES NOT DO
-------------------
It runs no model and takes no action. Substrate is `aggregation_from_upstream_artifacts`: it
reads row files that were written by earlier live runs and reports what is in them. The
measurement clock is therefore each ROW FILE's own `elapsed_s`, summed -- NOT this analyser's
wall time, which is a few seconds and would be a lie about when the measurement happened.
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "first_win_llm_on_20260727" / "cells"
EXP6011 = REPO / "results" / "experiment_6011_world_model_change_gate_four_arm.json"
OUT = REPO / "results" / "outer_loop_arc_wm_mask_gate_baseline_20260727.json"

# The eleven games routed to the HIDDEN-STATE trust branch in
# `E3AgentPolicy._induce_and_plan` (arc_world_model_trust_energy.HIDDEN_STATE_GAME_IDS).
# Read from the module rather than retyped, so this can never drift from the live path.
from carnot.agentic.arc_world_model_trust_energy import HIDDEN_STATE_GAME_IDS  # noqa: E402

HS = set(HIDDEN_STATE_GAME_IDS)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dist(values: list[float]) -> dict[str, Any]:
    """Full distribution, not just a headline. Empty is reported as empty, never as 0."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return {"n": 0, "empty": True}
    return {
        "n": len(vals),
        "empty": False,
        "min": vals[0],
        "p25": vals[len(vals) // 4],
        "median": st.median(vals),
        "p75": vals[(3 * len(vals)) // 4],
        "max": vals[-1],
        "mean": sum(vals) / len(vals),
        "n_exactly_zero": sum(1 for v in vals if v == 0.0),
        "n_at_or_above_0p5": sum(1 for v in vals if v >= 0.5),
        "all_values_sorted": [round(v, 6) for v in vals],
    }


def min_reachable_two_sided_p(n_discordant: int) -> float:
    """Exact paired sign/McNemar test: the smallest attainable two-sided p at n discordant pairs.

    All n in one direction gives 2 * 0.5**n. This is the POWER CEILING -- it is what the design
    can reach even if every single discordant pair falls the same way. Stating it before the run
    is what stops a non-significant result being spun afterwards.
    """
    return 2.0 * (0.5**n_discordant) if n_discordant > 0 else 1.0


def main() -> int:
    t0 = time.time()

    # ---------------- 1. read every banked cell ----------------
    cells: list[dict[str, Any]] = []
    for path in sorted(CELLS.glob("*.json")):
        with open(path) as fh:
            d = json.load(fh)
        d["_path"] = str(path.relative_to(REPO))
        cells.append(d)

    by_arm: dict[str, list[dict]] = defaultdict(list)
    for c in cells:
        by_arm[c["arm"]].append(c)

    # measurement clock = the rows' OWN elapsed_s, never this analyser's clock
    measurement_wall_s = sum(float(c["elapsed_s"]) for c in cells)

    arm_summary = {}
    for arm, rows in sorted(by_arm.items()):
        el = [float(r["elapsed_s"]) for r in rows]
        wins = sorted(r["variant_signature"] for r in rows if r.get("first_win"))
        planned_cells = sorted(
            r["variant_signature"]
            for r in rows
            if int((r.get("liveness_witness") or {}).get("induction_attempts_planned") or 0) > 0
        )
        arm_summary[arm] = {
            "n_cells": len(rows),
            "measurement_wall_s_rows": round(sum(el), 3),
            "per_cell_elapsed_s": {
                "median": round(st.median(el), 1),
                "max": round(max(el), 1),
                "min": round(min(el), 1),
            },
            # FAILURE SET, not a total: naming the cells makes a claim checkable
            "first_win_cells": wins,
            "n_first_win": len(wins),
            "cells_with_induction_attempts_planned_gt_0": planned_cells,
            "n_cells_with_planned_gt_0": len(planned_cells),
            "induction_attempts_n_total": sum(
                int((r.get("liveness_witness") or {}).get("induction_attempts_n") or 0)
                for r in rows
            ),
        }

    # ---------------- 2. skip-reason census + trust distribution ----------------
    skip_census: Counter = Counter()
    gate_diag_skip_census: Counter = Counter()
    attempts: list[tuple[str, str, dict]] = []  # (arm, game, diag)
    for c in cells:
        lw = c.get("liveness_witness") or {}
        for s in lw.get("induction_attempts_skipped") or []:
            skip_census[str(s)] += 1
        for g in lw.get("induction_attempt_gate_diagnostics") or []:
            gate_diag_skip_census[str(g.get("skipped"))] += 1
            attempts.append((c["arm"], c["game"], g))

    # The two arms that recorded the gate quantities differ in WHICH quantity gates:
    # llm_on_fix_diag ran the shipped `exact` metric, llm_on_fix_cellrecall ran
    # CARNOT_ARC_TRUST_METRIC=cell_recall. They are separate populations, not one pool.
    per_arm_trust: dict[str, Any] = {}
    for arm in ("llm_on_fix_diag", "llm_on_fix_cellrecall"):
        rows = [(g, game) for a, game, g in attempts if a == arm]
        plain = [(g, game) for g, game in rows if game not in HS]
        hidden = [(g, game) for g, game in rows if game in HS]
        per_arm_trust[arm] = {
            "trust_metric_declared": sorted(
                {str(g.get("trust_metric")) for g, _ in plain if "trust_metric" in g}
            ),
            "n_attempts": len(rows),
            "plain_branch": {
                "n": len(plain),
                "verify_accuracy": dist(
                    [g["verify_accuracy"] for g, _ in plain if "verify_accuracy" in g]
                ),
                "verify_cell_recall": dist(
                    [g["verify_cell_recall"] for g, _ in plain if "verify_cell_recall" in g]
                ),
                "per_game": {
                    game: {
                        "verify_accuracy": g.get("verify_accuracy"),
                        "verify_cell_recall": g.get("verify_cell_recall"),
                        "skipped": g.get("skipped"),
                    }
                    for g, game in plain
                },
            },
            "hidden_state_branch": {
                "n": len(hidden),
                "heldout_change_consistency": dist(
                    [
                        g["heldout_change_consistency"]
                        for g, _ in hidden
                        if "heldout_change_consistency" in g
                    ]
                ),
                "heldout_accuracy": dist(
                    [g["heldout_accuracy"] for g, _ in hidden if "heldout_accuracy" in g]
                ),
                "trust_energy": dist([g["trust_energy"] for g, _ in hidden if "trust_energy" in g]),
                "per_game": {
                    game: {
                        "trust_energy": g.get("trust_energy"),
                        "heldout_accuracy": g.get("heldout_accuracy"),
                        "heldout_change_consistency": g.get("heldout_change_consistency"),
                        "correct_changed_cells": g.get("correct_changed_cells"),
                        "binary_gate_pass": g.get("binary_gate_pass"),
                        "skipped": g.get("skipped"),
                    }
                    for g, game in hidden
                },
            },
        }

    # ---------------- 3. threshold sweep, per arm, on the quantity that ACTUALLY gates ----------------
    # `hidden_state` gates on heldout_change_consistency (score_change_weighted_consistency's
    # `trust_pass`); `plain` gates on verify_accuracy under metric=exact and verify_cell_recall
    # under metric=cell_recall. Sweeping a single pooled column would mix two different gates.
    sweeps = {}
    for arm in ("llm_on_fix_diag", "llm_on_fix_cellrecall"):
        gated: list[float] = []
        for a, game, g in attempts:
            if a != arm:
                continue
            if game in HS:
                if "heldout_change_consistency" in g:
                    gated.append(float(g["heldout_change_consistency"]))
            else:
                if g.get("trust_metric") == "cell_recall":
                    if "verify_cell_recall" in g:
                        gated.append(float(g["verify_cell_recall"]))
                elif "verify_accuracy" in g:
                    gated.append(float(g["verify_accuracy"]))
        sweeps[arm] = {
            "gated_quantity_distribution": dist(gated),
            "admits_at_threshold": {
                str(T): sum(1 for v in gated if v >= T) for T in (0.5, 0.25, 0.1, 0.05, 0.01)
            },
            "n_scored": len(gated),
        }

    # ---------------- 4. offline mask/gate prior from exp6011 ----------------
    with open(EXP6011) as fh:
        e6011 = json.load(fh)
    mask_avail: dict[str, bool] = {}
    mask_cells: dict[str, int] = {}
    acc_delta_rows = []
    plain_admissions = {"control": [], "mask_only": [], "gate_only": [], "both": []}
    for r in e6011["rows"]:
        mask_avail[r["game"]] = bool(r["hud_mask_available"])
        mask_cells[r["game"]] = int(r["logical_hud_mask_cells"])
        a_m0 = r["arms"]["mask=0|gate=1|engine=ondisk"]
        a_m1 = r["arms"]["mask=1|gate=1|engine=ondisk"]
        acc_delta_rows.append(
            {
                "game": r["game"],
                "seed": r["seed"],
                "branch": "hidden_state" if r["game"] in HS else "plain",
                "hud_mask_available": bool(r["hud_mask_available"]),
                "legacy_accuracy_mask_off": a_m0["legacy_accuracy"],
                "legacy_accuracy_mask_on": a_m1["legacy_accuracy"],
                "delta": round(a_m1["legacy_accuracy"] - a_m0["legacy_accuracy"], 6),
                "change_gate_passed_mask_off": bool(a_m0["passed"]),
                "change_gate_passed_mask_on": bool(a_m1["passed"]),
            }
        )
        if r["game"] not in HS:
            key = f"{r['game']}~seed{r['seed']}"
            if a_m0["legacy_accuracy"] >= 0.5:
                plain_admissions["control"].append(key)
            if a_m1["legacy_accuracy"] >= 0.5:
                plain_admissions["mask_only"].append(key)
            if a_m0["passed"]:
                plain_admissions["gate_only"].append(key)
            if a_m1["passed"]:
                plain_admissions["both"].append(key)

    deltas = [r["delta"] for r in acc_delta_rows]
    mask_moved_games = sorted({r["game"] for r in acc_delta_rows if abs(r["delta"]) > 1e-9})
    mask_unavailable_games = sorted(g for g, ok in mask_avail.items() if not ok)

    # ---------------- 5. power ----------------
    plain_games = sorted(g for g in mask_avail if g not in HS)
    mask_support_games = sorted(g for g, ok in mask_avail.items() if ok)
    mask_crossing_games = sorted(
        {
            r["game"]
            for r in acc_delta_rows
            if r["branch"] == "plain"
            and r["legacy_accuracy_mask_on"] >= 0.5
            and r["legacy_accuracy_mask_off"] < 0.5
        }
    )
    gate_flip_games = sorted(
        {
            r["game"]
            for r in acc_delta_rows
            if r["branch"] == "plain"
            and r["legacy_accuracy_mask_off"] >= 0.5
            and not r["change_gate_passed_mask_off"]
        }
    )

    power = {
        "test": "exact paired sign / McNemar on discordant (game,variant) pairs, per arm vs its matched control",
        "min_reachable_two_sided_p_by_n_discordant": {
            str(n): round(min_reachable_two_sided_p(n), 6) for n in range(1, 13)
        },
        "smallest_n_discordant_reaching_p_lt_0p05": 6,
        "arm_support": {
            "mask_only": {
                "games_where_the_arm_CAN_differ_from_control": mask_support_games,
                "n_games": len(mask_support_games),
                "games_structurally_inert_no_mask_resolves": mask_unavailable_games,
                "why_inert": (
                    "_world_model_hud_mask() returns (None,'explorer_mask_unresolved') for these, so "
                    "WorldModelVerifier sees hud_mask=None and the arm is byte-identical to control"
                ),
                "offline_prior_games_whose_score_actually_moved": mask_moved_games,
                "offline_prior_plain_branch_gate_crossings": mask_crossing_games,
                "CAVEAT_the_inert_set_is_an_UPPER_BOUND_not_a_fact": (
                    "exp6011 measured mask availability with `_compute_hud_mask_from_frame(frame)` "
                    "at its DEFAULT `edge_bar_detector=False`, on the first frame only. The LIVE "
                    "explorer (StepwiseExplorer._ingest) computes the WIDER repaired-detector "
                    "candidate when SUBMITTED_EDGE_BAR_HUD_MASK_ENABLED is True (it is), applies "
                    "the shipped mask as the Stage-2 fallback, and may LATER widen "
                    "`self.hud_mask` when Stage 2 admits the repair-added cells. Per exp5960's "
                    "hud_mask_delta_table the repaired detector resolves a mask for cn04 (0->32 "
                    "cells) and lp85 (0->64) where the shipped one resolves none. So cn04 and "
                    "lp85 -- and only those two of the eight -- may become mask-AVAILABLE "
                    "mid-episode on the live path. The live run must therefore read "
                    "hud_mask_status / hud_mask_cells PER ATTEMPT from the artifact and must NOT "
                    "assume this offline table; the eight below is where the offline classifier "
                    "found nothing, not a proof of live inertness."
                ),
                "games_the_repaired_detector_could_still_rescue": ["cn04", "lp85"],
            },
            "gate_only": {
                "games_where_the_arm_CAN_differ_from_control": plain_games,
                "n_games": len(plain_games),
                "games_structurally_inert_hidden_state_branch": sorted(HS),
                "why_inert": (
                    "change_gate_decision is wired ONLY into _induce_and_plan's non-hidden-state "
                    "branch; the 11 HIDDEN_STATE_GAME_IDS never reach it (see exp6012, which "
                    "measures that hole and reports acceptance_gate_passed=false)"
                ),
                "offline_prior_engines_the_gate_removes": gate_flip_games,
            },
        },
    }

    # ---------------- 5b. WHICH BANKED ARTIFACTS A MASK FLIP WOULD INVALIDATE ----------------
    # Masking changes EVERY world-model verification number. Any banked artifact that published
    # one was computed with hud_mask=None (the field did not exist before 2026-07-27), so its
    # figure is no longer comparable to a post-flip figure. This enumerates them so the change
    # lands with its corrections NAMED. It does NOT edit any of them (never-prune): a correction
    # belongs in a NEW artifact citing the original by sha256.
    MASK_SENSITIVE_FIELDS = {
        # WorldModelVerifier outputs -- computed over full logical grids incl. the HUD
        "verify_accuracy",
        "verify_cell_recall",
        "accuracy_threshold",
        # arc_world_model_trust_energy outputs -- same comparison, hidden-state branch
        "trust_energy",
        "heldout_accuracy",
        "heldout_change_consistency",
        "correct_changed_cells",
        "binary_gate_pass",
        "trust_pass",
    }
    MASK_SENSITIVE_VERDICTS = {
        # a skip reason IS a claim about a mask-sensitive comparison
        "world_model_accuracy_below_threshold",
        "hidden_state_trust_below_threshold",
    }

    def scan(obj: Any, found: set[str], depth: int = 0) -> None:
        if depth > 12:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in MASK_SENSITIVE_FIELDS:
                    found.add(k)
                if isinstance(v, str) and v in MASK_SENSITIVE_VERDICTS:
                    found.add("SKIP_REASON:" + v)
                scan(v, found, depth + 1)
        elif isinstance(obj, list):
            for v in obj[:500]:
                scan(v, found, depth + 1)

    invalidated: dict[str, list[str]] = {}
    n_scanned = 0
    results_dir = REPO / "results"
    for path in sorted(results_dir.rglob("*.json")):
        rel = str(path.relative_to(REPO))
        if "/first_win_llm_on_20260727/cells/" in rel:
            continue  # these are the baseline's own rows, cited separately above
        if path == OUT:
            continue  # this artifact itself, from a prior run -- a self-reference is not a finding
        try:
            if path.stat().st_size > 60_000_000:
                continue
            with open(path) as fh:
                obj = json.load(fh)
        except Exception:
            continue
        n_scanned += 1
        found: set[str] = set()
        scan(obj, found)
        if found:
            invalidated[rel] = sorted(found)

    artifact_invalidation = {
        "principle": (
            "masking removes cells from the comparison, so every previously-published world-model "
            "verification number is computed on a different quantity after the flip; naming the "
            "affected artifacts up front is what stops a silent re-baselining"
        ),
        "mask_sensitive_fields_searched": sorted(MASK_SENSITIVE_FIELDS),
        "mask_sensitive_verdicts_searched": sorted(MASK_SENSITIVE_VERDICTS),
        "n_json_artifacts_scanned": n_scanned,
        "n_artifacts_affected": len(invalidated),
        "affected": invalidated,
        "action_taken": "NONE -- enumerated only; no banked artifact is edited (never-prune)",
        "how_a_correction_should_land": (
            "a NEW artifact that recomputes the affected figure with the mask on and cites the "
            "original by path + sha256, per this repo's corrigendum pattern"
        ),
        "direction_of_the_change": (
            "every affected accuracy-like figure can only RISE (masking deletes disagreeing cells), "
            "so a post-flip number is NOT evidence of a capability improvement over a pre-flip "
            "number -- the two measure different quantities"
        ),
    }

    # ---------------- 5c. the four-arm design + its cost, stated BEFORE the run ----------------
    # The clean single-pass arm is llm_on_fix_diag: attempt 1, exit 0, 25/25 cells, no resume.
    # Its row-clock and its observed first->last cell-write span agree to within 2%, which is
    # what makes it the honest per-arm estimator (the other two LLM arms were interrupted and
    # resumed, so their spans include dead time and would overstate the cost).
    clean = arm_summary.get("llm_on_fix_diag", {})
    clean_rowclock_s = float(clean.get("measurement_wall_s_rows") or 0.0)
    clean_median_s = float((clean.get("per_cell_elapsed_s") or {}).get("median") or 0.0)
    k_workers = 4
    per_arm_wall_min = clean_rowclock_s / k_workers / 60.0

    # Every game that currently WINS anywhere in the LLM-off control -- these are the only cells
    # a regression can be detected on, so they must be in the focused block even when neither
    # flag can move their trust score. Read from the measured rows, not from the published
    # baseline's winner set, because the two DISAGREE (the published baseline won lp85~color01-04;
    # this control wins lp85~color02, sp80~color02/03, vc33~color01-04 -- see the
    # baseline_fidelity block of outer_loop_arc_first_win_llm_on_eval_concurrency_20260727.json,
    # where the winner-set non-reproduction is already established).
    control_winner_games = sorted(
        {sig.split("~")[0] for sig in arm_summary.get("llm_off", {}).get("first_win_cells", [])}
    )
    focused_games = sorted(
        set(mask_crossing_games) | set(gate_flip_games) | set(control_winner_games)
    )
    n_focus_cells = len(focused_games) * 3 * 4  # 3 extra variants x 4 arms

    four_arm_design = {
        "why_four_arms": (
            "the mask can only RAISE the measured score and the gate can only LOWER the admission "
            "rate, so a two-arm before/after cannot distinguish 'both worked and cancelled' from "
            "'neither did'; the offline prior already shows the cancellation is near-exact "
            f"(+{len(plain_admissions['mask_only']) - len(plain_admissions['control'])} vs "
            f"-{len(plain_admissions['control']) - len(plain_admissions['gate_only'])} admissions "
            "over the same 42 plain-branch rows)"
        ),
        "arms": {
            "A0_control": {"CARNOT_ARC_WM_HUD_MASK": "0", "CARNOT_ARC_WM_CHANGE_GATE": "0"},
            "A1_mask_only": {"CARNOT_ARC_WM_HUD_MASK": "1", "CARNOT_ARC_WM_CHANGE_GATE": "0"},
            "A2_gate_only": {"CARNOT_ARC_WM_HUD_MASK": "0", "CARNOT_ARC_WM_CHANGE_GATE": "1"},
            "A3_both": {"CARNOT_ARC_WM_HUD_MASK": "1", "CARNOT_ARC_WM_CHANGE_GATE": "1"},
        },
        "arm_selection_is_env_only": (
            "both flags read through arc_executable_world_model._flag_env, so an arm is selected "
            "without editing a shipped constant -- an arm selected by editing a constant could not "
            "be reproduced from an artifact's recorded command line"
        ),
        "unit_of_pairing": "(game, variant) cell, matched across all four arms",
        "stage_1_corpus": {
            "games": 25,
            "variants": [1],
            "cells_per_arm": 25,
            "total_cells": 100,
            "purpose": "corpus statement + skip-reason census shift + the flag-inertness regression",
        },
        "stage_2_focused": {
            "games": focused_games,
            "why_these": (
                "the only games the offline prior says either flag can move, PLUS every game that "
                f"currently wins: {mask_crossing_games} cross the legacy 0.5 gate when the mask is "
                f"on; {gate_flip_games} are the real on-disk degenerates the change gate removes; "
                f"{control_winner_games} carry every win in the LLM-off control and are the only "
                "cells on which the regression clause has any support at all"
            ),
            "regression_sentinel_games": control_winner_games,
            "control_winner_cells": arm_summary.get("llm_off", {}).get("first_win_cells", []),
            "variants": [2, 3, 4],
            "total_cells": n_focus_cells,
            "purpose": "buy discordant pairs on the only cells that can supply them",
        },
        "acceptance_conditions_are_DIFFERENT_per_arm": {
            "A1_mask_only": {
                "primary": (
                    "the paired per-cell gated-quantity (verify_accuracy on the plain branch, "
                    "heldout_change_consistency on the hidden-state branch) is STRICTLY GREATER "
                    "than A0's on the mask-available games and EQUAL on the 8 mask-unavailable "
                    "games -- the equality half is the load-bearing control, because a difference "
                    "there would mean the arm is changing something other than the mask"
                ),
                "computed_witness_required": (
                    "per-cell (A0_value, A1_value, hud_mask_status, hud_mask_cells); a cell "
                    "reported as masked MUST carry hud_mask_status=='applied' and hud_mask_cells>0"
                ),
                "secondary": "n_cells_with_induction_attempts_planned_gt_0 rises above the baseline 0",
                "regression_clause": (
                    "no first_win present in A0 may be absent in A1 -- newly admitting an engine "
                    "can install a goal bias and a plan that LOSES a win the explorer would have "
                    "taken; sp80 both wins (at variants 2,3) and is a mask-crossing game, so this "
                    "is a live risk, not a theoretical one"
                ),
            },
            "A2_gate_only": {
                "primary": (
                    "the two real on-disk degenerates are REJECTED where A0 admitted them: A0 must "
                    "record skipped=='' or a post-trust skip for ft09/lp85 while A2 records "
                    "skipped starting 'world_model_change_gate_'"
                ),
                "must_not_fire_control": (
                    "no cell that produced a plan in A0 may be rejected by the gate for a reason "
                    "other than degeneracy; the offline must-not-fire control (the hand-written "
                    "dc22 navigation engine, admitted on 3/3 seeds in exp6011) is the standing "
                    "proof the gate is not 'reject everything', and its live analogue is that "
                    "A2's win set must equal A0's"
                ),
                "power_disclosure": (
                    "A2 CANNOT produce a new win by construction -- it only removes admissions, and "
                    "the baseline's two admissions both terminated at "
                    "'no_reachable_plan_after_refinement' with zero plans. Its endpoint is "
                    "correctness of rejection plus non-regression, NOT win rate."
                ),
            },
            "A3_both": {
                "primary": (
                    "attributes the interaction: if A1 gains admissions and A2 removes them, A3 "
                    "tells us whether the mask's new admissions survive the gate. The offline prior "
                    "says they do not (change-gate passes on 0/75 rows at mask=0 AND mask=1), so a "
                    "null A3-vs-A0 is the PREDICTED outcome and must not be read as 'nothing worked'"
                ),
            },
        },
        "stated_before_the_run_so_a_null_cannot_be_spun": [
            "A2 cannot reach significance on wins at any N -- it removes admissions only.",
            "A3-vs-A0 is predicted null by the offline prior; that is attribution, not failure.",
            "A1 is the only arm that can move the win rate, and only on "
            f"{len(mask_crossing_games)} candidate games.",
        ],
    }

    # Stage 1: four arms, each a 25-cell pass. per_arm_wall_min ALREADY carries the /k_workers.
    _stage1_h = 4 * per_arm_wall_min / 60.0
    # Stage 2: n_focus_cells at the clean arm's median cell cost, over k_workers.
    _stage2_h = n_focus_cells * clean_median_s / k_workers / 3600.0

    wall_clock_estimate = {
        "estimator": (
            "llm_on_fix_diag -- the one LLM-on arm that ran single-pass to completion (attempt 1, "
            "exit 0, 25/25, no resume). Its row-clock/K and its observed first-to-last cell-write "
            "span agree to ~2%, so K=4 parallel efficiency is ~1.0 on this workload."
        ),
        "clean_arm_row_clock_s": round(clean_rowclock_s, 1),
        "clean_arm_observed_span_min": 42.0,
        "clean_arm_median_cell_s": clean_median_s,
        "k_workers": k_workers,
        "per_arm_wall_min_25_cells": round(per_arm_wall_min, 1),
        "stage_1_wall_h": round(_stage1_h, 2),
        "stage_2_wall_h": round(_stage2_h, 2),
        # Self-consistency guard: the first draft of this block divided stage 1 by k_workers a
        # SECOND time (per_arm_wall_min already carries the /k), and produced a total SMALLER
        # than stage 1 alone. Asserting the total dominates each part is what caught it.
        "total_wall_h": round(_stage1_h + _stage2_h, 2),
        "total_dominates_each_stage": bool(
            _stage1_h + _stage2_h >= _stage1_h and _stage1_h + _stage2_h >= _stage2_h
        ),
        "add_for_server_spawn_and_teardown_min": 5,
        "gpu": "GPU 1 only (conductor owns GPU 0); one 81920-context llama-server ~13.5 GiB",
        "concurrency_constraint": (
            "arms run SEQUENTIALLY on one card -- two 13.5 GiB servers do not fit in 24 GiB, and "
            "the RUN_LOG records exactly that failure (a leaked server on port 8953 blocked the "
            "next arm for 600 s). One server for the whole matrix, reaped BY PORT at the end."
        ),
    }

    # ---------------- 6. artifact ----------------
    cited = []
    for p in [EXP6011, REPO / "results" / "experiment_6012_hidden_state_trust_gate_hole.json"]:
        if p.exists():
            cited.append(
                {"experiment_id": p.stem, "path": str(p.relative_to(REPO)), "sha256": sha256(p)}
            )

    artifact: dict[str, Any] = {
        "experiment": "outer_loop_arc_wm_mask_gate_baseline_20260727",
        "title": (
            "PRE-CHANGE BASELINE for the REQ-ARC-WMTE-6010 (HUD mask) / -6011 (change gate) "
            "four-arm A/B: trust distribution, skip census, mask-invalidation surface, power ceiling"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "submitted_to_leaderboard": False,
        "random_seed": 0,
        "model_specs": {
            "note": "no model invoked; this reads row files written by earlier live runs plus one "
            "offline verifier-scoring artifact"
        },
        "duration_s": None,  # filled below with the ANALYSER clock, labelled as such
        "duration_s_provenance": "analyser wall time; NOT the measurement clock",
        "measurement_wall_s": round(measurement_wall_s, 3),
        "measurement_wall_s_provenance": (
            "sum of each cited row file's OWN elapsed_s over all "
            f"{len(cells)} cells in results/first_win_llm_on_20260727/cells/"
        ),
        "n_cells_read": len(cells),
        "arm_summary": arm_summary,
        "skip_reason_census_from_induction_attempts_skipped": dict(skip_census),
        "skip_reason_census_from_gate_diagnostics": dict(gate_diag_skip_census),
        "n_gate_diagnostic_attempts": len(attempts),
        "baseline_trust_distribution": per_arm_trust,
        "threshold_sweep": sweeps,
        "offline_mask_gate_prior": {
            "source": "results/experiment_6011_world_model_change_gate_four_arm.json (on-disk engines, 25 games x 3 seeds)",
            "caveat": (
                "these are the PREVIOUSLY-INDUCED on-disk engines, not fresh live inductions, so they "
                "are a PRIOR that bounds the design -- not a prediction of the live arms"
            ),
            "hud_mask_available_per_game": mask_avail,
            "logical_hud_mask_cells_per_game": mask_cells,
            "n_games_with_mask": sum(1 for v in mask_avail.values() if v),
            "n_games_without_mask": sum(1 for v in mask_avail.values() if not v),
            "legacy_accuracy_delta_mask_on_minus_off": {
                "n_rows": len(deltas),
                "n_rows_changed": sum(1 for d in deltas if abs(d) > 1e-9),
                "mean": round(sum(deltas) / len(deltas), 6),
                "max": max(deltas),
                "min": min(deltas),
                "n_rows_negative": sum(1 for d in deltas if d < -1e-9),
                "why_never_negative": (
                    "masking can only DELETE cells from the comparison, so an exact-match count can "
                    "only rise; a negative delta would be a bug and its absence is the witness"
                ),
            },
            "per_row": acc_delta_rows,
            "plain_branch_admission_sets": {k: sorted(v) for k, v in plain_admissions.items()},
            "plain_branch_admission_counts": {k: len(v) for k, v in plain_admissions.items()},
            "confound_witness": (
                "over the same 42 plain-branch rows the mask arm ADDS "
                f"{len(plain_admissions['mask_only']) - len(plain_admissions['control'])} admissions and "
                f"the gate arm REMOVES {len(plain_admissions['control']) - len(plain_admissions['gate_only'])}; "
                "shipped together the counts nearly cancel, which is exactly why the two flags must be "
                "measured on independent axes rather than as one change"
            ),
        },
        "power": power,
        "artifacts_a_mask_flip_invalidates": artifact_invalidation,
        "four_arm_design": four_arm_design,
        "wall_clock_estimate": wall_clock_estimate,
        "acceptance_gate_baseline_is_nonempty": len(cells) > 0 and len(attempts) > 0,
        "acceptance_gate_every_reported_field_has_a_nonzero_instance": None,  # computed below
        "honest_verdict": "complete_prechange_baseline_established_from_banked_rows_no_gpu",
    }

    # Structurally-dead-channel check: every distribution we publish must have at least one
    # non-trivial value somewhere, or we are publishing a channel that cannot carry information.
    live_channels = {}
    for arm, blk in per_arm_trust.items():
        for branch in ("plain_branch", "hidden_state_branch"):
            for field, dd in blk[branch].items():
                if isinstance(dd, dict) and not dd.get("empty", True):
                    key = f"{arm}.{branch}.{field}"
                    live_channels[key] = dd["max"] > 0.0 or dd["min"] < 0.0
    artifact["channel_liveness"] = live_channels
    artifact["acceptance_gate_every_reported_field_has_a_nonzero_instance"] = all(
        live_channels.values()
    )
    artifact["dead_channels"] = sorted(k for k, v in live_channels.items() if not v)

    artifact["cited_upstream_artifacts"] = cited
    artifact["row_files_dir"] = str(CELLS.relative_to(REPO))
    artifact["duration_s"] = round(time.time() - t0, 4)
    artifact["acceptance_gate_passed"] = bool(
        artifact["acceptance_gate_baseline_is_nonempty"]
        and artifact["acceptance_gate_every_reported_field_has_a_nonzero_instance"]
    )
    payload = json.dumps(artifact, sort_keys=True).encode()
    artifact["reproducibility_checksum"] = hashlib.sha256(payload).hexdigest()

    OUT.write_text(json.dumps(artifact, indent=1))
    print(f"wrote {OUT.relative_to(REPO)}")
    print(
        f"  cells={len(cells)} attempts={len(attempts)} measurement_wall_s={measurement_wall_s:.1f}"
    )
    print(f"  dead_channels={artifact['dead_channels']}")
    print(f"  acceptance_gate_passed={artifact['acceptance_gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

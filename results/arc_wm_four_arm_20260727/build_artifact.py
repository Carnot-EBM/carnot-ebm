#!/usr/bin/env python
"""Build results/experiment_6015_wm_hud_mask_change_gate_four_arm_live.json.

The verdict, the acceptance gates and every headline number are COMPUTED from
analysis_fourarm.json (which is itself a projection of the live row files). Nothing here is
typed in by hand: a hand-typed headline is a number that cannot be traced to an artifact,
which is exactly what G4 exists to prevent.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "results" / "arc_wm_four_arm_20260727"))
OUT = REPO / "results" / "arc_wm_four_arm_20260727"

import analyse_fourarm as A  # noqa: E402, N812

EXP = 6015
ART = REPO / "results" / f"experiment_{EXP}_wm_hud_mask_change_gate_four_arm_live.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    t0 = time.time()
    a = A.analyse()
    (OUT / "analysis_fourarm.json").write_text(json.dumps(a, indent=1, default=str))

    per = a["per_arm"]
    cmp_ = a["comparisons_vs_control"]
    wit = a["witnesses"]
    integ = a["arm_integrity"]
    n = a["n_matched_cells_per_arm"]

    # ---------------- acceptance gates, each with a COMPUTED witness -------------
    # Every gate below has a reachable failing value on this data. A gate whose pass
    # region is the whole space is not a gate.
    g_arms_took = all(integ[k]["resolver_matches_declaration"] for k in A.ARMS)
    g_follow = all(integ[k]["follow_default_took"] is True for k in ("wm_A2_gate", "wm_A3_both"))
    g_device = all((integ[k]["server_device"] or "").startswith("CONFIRMED_GPU") for k in A.ARMS)
    g_no_card_drop = all(
        not ((integ[k]["vram_summary"] or {}).get("dropped_below_1gib_mid_run")) for k in A.ARMS
    )
    g_generator_live = all(wit["generator_liveness"][k]["generator_answered"] for k in A.ARMS)
    g_matched = n > 0 and not a["unmatched_signatures"]
    g_mask_proof = wit["mask_application_proof"]["all_claims_proved"]
    g_inert_equal = wit["mask_inert_equality_control"]["holds"]
    g_no_regression = all(cmp_[k]["first_win"]["regression_clause_holds"] for k in A.TREATMENTS)
    # NOT a pass condition -- recorded so a fully-inert matrix is impossible to mistake for
    # a measured null (the 174/174 bit-identical failure).
    not_all_identical = any(cmp_[k]["n_bit_identical_to_control"] < n for k in A.TREATMENTS)

    # ---------------- VACUITY DISCLOSURE ----------------------------------------
    # Four of the gates above are universally-quantified over a set that can be EMPTY, and
    # `all([])` is True. A gate that passes because its pass region is empty is a PASS THAT
    # COULD NOT HAVE FAILED -- the exact trap the REQ-6013 work hit, where ft09's rejection
    # was counted as a catch until its held-out split turned out to contain 0 of 40 changing
    # transitions. Each vacuity flag below is computed from the same data the gate reads, and
    # a vacuous gate is reported as vacuous rather than silently banked as evidence.
    n_mask_claims = wit["mask_application_proof"]["n_claims"]
    n_inert_checked = wit["mask_inert_equality_control"]["n_cells_checked"]
    n_control_wins = per[A.CONTROL]["n_first_win"]
    n_vram_summaries = sum(1 for k in A.ARMS if integ[k]["vram_summary"])
    vacuity = {
        "acceptance_gate_every_mask_claim_carries_positive_cell_count": {
            "vacuous": n_mask_claims == 0,
            "support": n_mask_claims,
            "meaning_if_vacuous": (
                "the mask arm applied a mask on ZERO cells -- the gate passes trivially AND "
                "the mask treatment never reached the corpus, which is itself the finding"
            ),
        },
        "acceptance_gate_mask_arm_identical_to_control_where_no_mask_resolved": {
            "vacuous": n_inert_checked == 0,
            "support": n_inert_checked,
            "meaning_if_vacuous": "no matched cells at all, so nothing constrained the control",
        },
        "acceptance_gate_no_control_win_lost_by_any_treatment_arm": {
            "vacuous": n_control_wins == 0,
            "support": n_control_wins,
            "meaning_if_vacuous": (
                "the control won on ZERO cells, so no treatment arm COULD have lost a win; "
                "the regression clause has no support and must not be read as reassurance"
            ),
        },
        "acceptance_gate_no_card_dropped_off_bus_mid_run": {
            "vacuous": n_vram_summaries < len(A.ARMS),
            "support": n_vram_summaries,
            "meaning_if_vacuous": (
                "at least one arm recorded no VRAM residency timeseries, so a mid-run card "
                "drop in that arm would be undetected rather than absent"
            ),
        },
    }
    n_vacuous = sum(1 for v in vacuity.values() if v["vacuous"])

    gates = {
        "acceptance_gate_all_four_arms_resolved_as_declared": g_arms_took,
        "acceptance_gate_hidden_state_follow_default_observed": g_follow,
        "acceptance_gate_device_confirmed_by_per_pid_residency": g_device,
        "acceptance_gate_no_card_dropped_off_bus_mid_run": g_no_card_drop,
        "acceptance_gate_generator_answered_in_every_arm": g_generator_live,
        "acceptance_gate_every_cell_matched_across_all_four_arms": g_matched,
        "acceptance_gate_every_mask_claim_carries_positive_cell_count": g_mask_proof,
        "acceptance_gate_mask_arm_identical_to_control_where_no_mask_resolved": g_inert_equal,
        "acceptance_gate_no_control_win_lost_by_any_treatment_arm": g_no_regression,
    }
    passed = all(gates.values())

    # ---------------- the two headline questions --------------------------------
    mask_plain_ctrl = per[A.CONTROL]["gated_quantity_plain_verify_accuracy"]
    mask_plain_a1 = per["wm_A1_mask"]["gated_quantity_plain_verify_accuracy"]
    mask_hidden_ctrl = per[A.CONTROL]["gated_quantity_hidden_heldout_accuracy"]
    mask_hidden_a1 = per["wm_A1_mask"]["gated_quantity_hidden_heldout_accuracy"]

    moved_off_floor = bool(
        mask_plain_a1.get("median") is not None
        and float(mask_plain_a1.get("median") or 0.0) > float(mask_plain_ctrl.get("median") or 0.0)
    )
    any_score_moved = cmp_["wm_A1_mask"]["n_moved"] > 0
    admissions_rose = per["wm_A1_mask"]["n_planned_gt_0"] > per[A.CONTROL]["n_planned_gt_0"]

    gate_rej = wit["gate_rejections"]
    gate_rejects_degenerates = (
        gate_rej["n"] > 0 and gate_rej["n_justified_low_fidelity"] == gate_rej["n"]
    )
    gate_kept_wins = cmp_["wm_A2_gate"]["first_win"]["regression_clause_holds"]
    gate_rejects_everything = (
        per["wm_A2_gate"]["n_planned_gt_0"] == 0 and per[A.CONTROL]["n_planned_gt_0"] > 0
    )

    headline = {
        "mask_arm_question": "does the trust-score distribution MOVE off the floor?",
        "mask_arm_answer": {
            "plain_branch_control_distribution": mask_plain_ctrl,
            "plain_branch_mask_distribution": mask_plain_a1,
            "hidden_branch_control_distribution": mask_hidden_ctrl,
            "hidden_branch_mask_distribution": mask_hidden_a1,
            "n_cells_whose_gated_quantity_moved": cmp_["wm_A1_mask"]["n_moved"],
            "median_moved_off_floor": moved_off_floor,
            "any_score_moved": any_score_moved,
            "admissions_rose_above_control": admissions_rose,
            "sign_test": cmp_["wm_A1_mask"]["gated_quantity_sign_test"],
            # The plain-language reading, stated whichever way the data falls. A null here
            # is a REAL and VALUABLE answer -- it would mean the wall is genuine capability
            # and not a measurement artifact -- so it must be said plainly, not buried.
            "measurement_artifact_hypothesis": (
                "SUPPORTED_IN_PART: masking moved the measured score on at least one cell"
                if any_score_moved
                else "REFUTED: masking moved the measured score on ZERO cells; on this corpus "
                "the median-0.0 trust score is NOT a HUD measurement artifact, and the "
                "induced-world-model wall is genuine capability"
            ),
        },
        "gate_arm_question": "are degenerates rejected AND good engines kept?",
        "gate_arm_answer": {
            "n_gate_rejections": gate_rej["n"],
            "n_rejections_justified_by_low_change_fidelity": gate_rej["n_justified_low_fidelity"],
            "n_rejections_where_incumbent_gate_would_have_PASSED": gate_rej[
                "n_where_incumbent_would_have_passed"
            ],
            "rejects_degenerates": gate_rejects_degenerates,
            "kept_every_control_win": gate_kept_wins,
            "rejects_everything_degenerate_gate": gate_rejects_everything,
            "admission_control": per[A.CONTROL]["n_planned_gt_0"],
            "admission_gate_arm": per["wm_A2_gate"]["n_planned_gt_0"],
            "both_directions_reported": True,
        },
    }

    verdict = (
        "complete_four_arm_live_matrix_measured_"
        + ("mask_moved_scores_" if any_score_moved else "mask_moved_nothing_")
        + (
            f"gate_rejected_{gate_rej['n']}_degenerate_engines"
            if gate_rej["n"]
            else "gate_rejected_none"
        )
    )

    art = {
        "experiment": EXP,
        "experiment_id": EXP,
        "title": (
            "REQ-ARC-WMTE-6010/-6011/-6013 four-arm LIVE matrix: HUD-mask and change-gate, "
            "control / mask-only / gate-only / both, per-cell matched"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "milestone": "outer-loop 2026-07-27",
        # The analyser reads persisted row files and republishes their own elapsed_s; it
        # loads no model and runs no inference. Declaring live_llm_inference here would be
        # false even though the ROWS were produced by live inference.
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_note": (
            "the ROWS were produced by a live GPU-1 Qwen3.5-9B-MTP generator (per-arm server "
            "launch + per-PID residency recorded in arm_integrity); THIS artifact is the "
            "analyser pass over those persisted rows, so measurement_wall_s comes from the "
            "rows' own elapsed_s and duration_s is the analyser clock only"
        ),
        "verifier_is_oracle": False,
        "verifier_is_oracle_note": (
            "no moat/efficiency claim is made here. The quantity under test is a world-model "
            "verifier's agreement with OBSERVED env transitions, used as an admission gate; "
            "this experiment measures whether that gate admits and rejects the right engines, "
            "not whether an energy verifier beats a baseline"
        ),
        "solve_provenance": "development_proxy",
        "solve_provenance_note": (
            "public games via the shipped E3AgentPolicy through experiment_4605's real "
            "run_variant_attempt; NO level solve is claimed by this artifact"
        ),
        "submitted_to_leaderboard": False,
        "offline_reproduced": False,
        "reproduced_levels": 0,
        "random_seed": 0,
        "random_seeds_used": [0],
        "random_seed_note": (
            "the pairing unit is the (game, variant) cell, not an RNG seed: all four arms run "
            "the SAME 25 games at variant 1 and are compared per cell. exp4605's variant "
            "specs are deterministic given the game list, so the cell set is the replicate "
            "structure and it is matched by construction"
        ),
        "model_specs": {
            "generator": "unsloth/Qwen3.5-9B-MTP-GGUF (Qwen3.5-9B-Q4_K_M.gguf)",
            "n_ctx": 81920,
            "kv_quant": "q8_0",
            "max_tokens": 4096,
            "server": "~/.cache/llama.cpp-master/build/bin/llama-server (CUDA build)",
            "device": "GPU 1 (RTX 3090), confirmed per-arm by per-PID VRAM residency",
            "note": (
                "the 9B, not the 31B: a 21GB gemma-4-31B on a 24GB card triggered the "
                "documented eGPU PCI-bus fault on 2026-07-24 (GPU1 dropped 21GB->4MiB "
                "mid-run). The 9B measures ~13.5 GiB at n_ctx=81920"
            ),
        },
        "duration_s": round(time.time() - t0, 4),
        "duration_s_provenance": "analyser wall time; NOT the measurement clock",
        "measurement_wall_s": a["measurement_wall_s"],
        "measurement_wall_s_provenance": (
            "sum of every cited cell row file's OWN elapsed_s across all four arms"
        ),
        "preconditions_checked": [
            {"resource": "GPU 1 free (conductor stopped by operator)", "available": True},
            {"resource": "Qwen3.5-9B-MTP GGUF cached", "available": True},
            {"resource": "llama.cpp CUDA llama-server binary", "available": True},
            {"resource": "25 public games in environment_files/", "available": True},
            {
                "resource": "per-arm device residency CONFIRMED_GPU1_BY_PER_PID_RESIDENCY",
                "available": g_device,
            },
        ],
        "four_arms": A.ARM_DECLARED,
        "arm_integrity": integ,
        "per_arm": per,
        "comparisons_vs_control": cmp_,
        "interaction": a["interaction"],
        "headline": headline,
        "witnesses": wit,
        "matched_signatures": a["matched_signatures"],
        "unmatched_signatures": a["unmatched_signatures"],
        "n_matched_cells_per_arm": n,
        "not_all_arms_bit_identical_to_control": not_all_identical,
        "field_provenance": {
            "measurement_wall_s": {
                "principle": (
                    "the analyser clock is not the measurement clock; republishing each row's "
                    "own elapsed_s is what makes a 6-second analyser pass over a 4-hour "
                    "measurement honest instead of a DURATION_TOO_SHORT fabrication signal"
                ),
                "satisfied_by": "sum of cells/*.json elapsed_s",
            },
            "arm_integrity.resolved": {
                "principle": (
                    "setting an env var and asserting the arm is configured is the "
                    "declared-vs-actual gap; reading the resolver back through the SAME "
                    "helper the agent calls is the witness that the arm actually took"
                ),
                "satisfied_by": "world_model_*_enabled() read at run time, stored in run_*.json",
            },
            "witnesses.mask_inert_equality_control": {
                "principle": (
                    "the equality half is the load-bearing control: a difference on cells "
                    "where no mask resolved would mean the arm changes something other than "
                    "the mask, making every mask-arm number uninterpretable"
                ),
                "satisfied_by": "per-cell gated-quantity comparison on hud_mask_status != applied",
            },
            "comparisons_vs_control.n_bit_identical_to_control": {
                "principle": (
                    "the 2026-07-27 first-win measurement reported p=1.0 on 74/74 cells that "
                    "were BIT-IDENTICAL to their controls -- an arithmetic identity, not a "
                    "measurement; counting identity up front makes that unmissable"
                ),
                "satisfied_by": (
                    "per-cell equality on first_win/actions/reached_level/planned/skipped"
                ),
            },
            "headline.mask_arm_answer.measurement_artifact_hypothesis": {
                "principle": (
                    "a REFUTED artifact-hypothesis is a real and valuable answer (it means "
                    "the wall is genuine capability), so it must be stated plainly rather "
                    "than presented as a disappointing null"
                ),
                "satisfied_by": "computed from n_cells_whose_gated_quantity_moved",
            },
            "sign_test.min_reachable_two_sided_p": {
                "principle": (
                    "with n discordant pairs the smallest two-sided p is 2^(1-n); reporting "
                    "it stops an underpowered null being read as evidence of absence"
                ),
                "satisfied_by": "2**(1-n_discordant), reported with underpowered_by_construction",
            },
        },
        **gates,
        "acceptance_gate_passed": passed,
        "unmet_gates": sorted(k for k, v in gates.items() if not v),
        # A vacuous gate is NOT counted as evidence. Listed separately from unmet_gates
        # because it is a third state: not failed, but not earned either.
        "gate_vacuity_disclosure": vacuity,
        "n_vacuous_gates": n_vacuous,
        "vacuous_gates": sorted(k for k, v in vacuity.items() if v["vacuous"]),
        "acceptance_gate_passed_and_none_vacuous": bool(passed and n_vacuous == 0),
        "honest_verdict": verdict,
        "cited_upstream_artifacts": [
            {
                "path": "results/outer_loop_arc_wm_mask_gate_baseline_20260727.json",
                "role": "pre-change baseline + four-arm design + power table",
                "sha256": _sha(
                    REPO / "results" / "outer_loop_arc_wm_mask_gate_baseline_20260727.json"
                ),
            },
            {
                "path": "results/experiment_6013_hidden_state_change_gate_closure.json",
                "role": "the REQ-6013 hidden-state gate closure this run exercises live",
                "sha256": _sha(
                    REPO / "results" / "experiment_6013_hidden_state_change_gate_closure.json"
                ),
            },
        ],
        "row_files_dir": "results/arc_wm_four_arm_20260727/cells/",
        "runner": "results/arc_wm_four_arm_20260727/fourarm.py",
        "analyser": "results/arc_wm_four_arm_20260727/analyse_fourarm.py",
        "git_head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in art.items() if k not in ("run_date", "duration_s")},
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()
    ART.write_text(json.dumps(art, indent=1, default=str))
    print("wrote", ART)
    print("verdict:", verdict)
    print("gates:", json.dumps(gates, indent=1))
    print("acceptance_gate_passed:", passed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
